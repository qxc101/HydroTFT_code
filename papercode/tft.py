"""Temporal Fusion Transformer (TFT) implementations for hydrological modeling.

Reference:
Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal fusion
transformers for interpretable multi-horizon time series forecasting.
International Journal of Forecasting, 37(4), 1748-1764.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstm import LSTM as CustomLSTM




class GLU(nn.Module):
    """Gated Linear Unit: projects to 2x width, then applies sigmoid gating."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.fc(x), dim=-1)


class GRN(nn.Module):
    """Gated Residual Network with optional context input.

    Implements:
        eta2 = ELU(W2 * a  +  W3 * c  +  b2)
        eta1 = W1 * eta2  +  b1
        GRN(a, c) = LayerNorm(GLU(eta1) + skip(a))
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.1,
                 context_size: Optional[int] = None):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(output_size)
        self.layer_norm = nn.LayerNorm(output_size)

        self.context_fc = (nn.Linear(context_size, hidden_size, bias=False)
                           if context_size is not None else None)
        self.skip_fc = (nn.Linear(input_size, output_size)
                        if input_size != output_size else None)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip_fc(x) if self.skip_fc is not None else x

        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        gated = self.glu(hidden)

        return self.layer_norm(gated + residual)


class GateAddNorm(nn.Module):
    """Gated skip connection: LayerNorm(Dropout(GLU(x)) + skip)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GLU(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.dropout(self.glu(x)) + skip)



class VariableSelectionNetwork(nn.Module):
    """Vectorized Variable Selection Network.

    Replaces the standard per-variable loop-based VSN with GPU-efficient
    broadcasting operations. Each variable is independently embedded via a
    parameter matrix (equivalent to per-variable Linear(1, d_model)), and
    importance weights are computed via a lightweight scoring network with
    cross-variable interaction through mean-pooled embeddings.

    Architecture:
        1. Per-variable embedding via broadcasting (no loops)
        2. Variable importance scoring with cross-variable interaction
        3. Optional static context conditioning on weights
        4. Post-aggregation GRN for nonlinear transformation
    """

    def __init__(self,
                 n_vars: int,
                 d_model: int,
                 dropout: float = 0.1,
                 context_size: Optional[int] = None):
        super().__init__()

        self.n_vars = n_vars
        self.d_model = d_model

        # Per-variable embedding: equivalent to n_vars separate Linear(1, d_model)
        # but implemented as a single broadcast multiply + add.
        self.emb_weight = nn.Parameter(torch.empty(n_vars, d_model))
        self.emb_bias = nn.Parameter(torch.empty(n_vars, d_model))
        # Match nn.Linear(1, d_model) initialization
        nn.init.kaiming_uniform_(self.emb_weight, a=math.sqrt(5))
        nn.init.uniform_(self.emb_bias, -1.0, 1.0)

        # Variable importance scoring (memory-efficient alternative to
        # concatenating all embeddings into a huge [n_vars*d_model] vector).
        # Per-variable score from its own embedding:
        self.var_score = nn.Linear(d_model, 1, bias=True)
        # Cross-variable interaction via mean-pooled embedding summary:
        self.cross_score = nn.Linear(d_model, n_vars, bias=False)
        # Optional static context conditioning:
        self.ctx_gate = (nn.Linear(context_size, n_vars, bias=False)
                         if context_size is not None else None)

        # Post-aggregation GRN (replaces per-variable GRN loop with a single
        # GRN applied after the weighted sum of embeddings).
        self.output_grn = GRN(d_model, d_model, d_model, dropout)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [..., n_vars] where each feature is a scalar.
        context : torch.Tensor, optional
            Context vector from static covariate encoder, shape [..., d_model]
            or [..., context_size].

        Returns
        -------
        selected : torch.Tensor
            Weighted combination of variable representations, shape [..., d_model].
        weights : torch.Tensor
            Variable importance weights (softmax), shape [..., n_vars].
        """
        # Per-variable embedding via broadcasting (GPU-parallel, no loops).
        emb = x.unsqueeze(-1) * self.emb_weight + self.emb_bias  # [..., n_vars, d_model]

        # Variable importance scoring:
        scores = self.var_score(emb).squeeze(-1)  # [..., n_vars]
        pool = emb.mean(dim=-2)  # [..., d_model]
        scores = scores + self.cross_score(pool)  # [..., n_vars]
        if self.ctx_gate is not None and context is not None:
            scores = scores + self.ctx_gate(context)
        weights = F.softmax(scores, dim=-1)  # [..., n_vars]

        # Weighted aggregation of per-variable embeddings
        selected = (emb * weights.unsqueeze(-1)).sum(dim=-2)  # [..., d_model]

        # Post-aggregation nonlinear transformation
        selected = self.output_grn(selected)

        return selected, weights



class InterpretableMultiHeadAttention(nn.Module):
    """TFT-style attention with shared value projection for interpretability.

    Separate Q/K projections are split into per-head subspaces, but the value
    projection is shared across all heads. Attention weights are averaged across
    heads before being applied to the shared values, producing a single
    interpretable attention pattern.

    InterpretableMultiHead(Q, K, V) = (1/H * sum_h A_h) * V * W_V  then * W_O
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q/K are split per-head; V is shared
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.size()

        # Per-head Q and K
        Q = self.w_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Shared V (full d_model, NOT split by head)
        V = self.w_v(value)  # [B, T, d_model]

        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))  # [B, H, T, T]

        # Average attention across heads, apply to shared V
        attn_avg = attn.mean(dim=1)  # [B, T, T]
        output = self.w_o(torch.matmul(attn_avg, V))  # [B, T, d_model]

        return output, attn


class TFT(nn.Module):
    """Full Temporal Fusion Transformer (Lim et al., 2021).

    Architecture flow:
        Static attrs  -->  Static VSN  -->  4 context GRNs (c_s, c_e, c_c, c_h)
                                                    |
        Dynamic feats -->  Temporal VSN (with c_s)  -->  LSTM (init h0=c_e, c0=c_c)
                                                    |
                            GateAddNorm(lstm_out, vsn_out)
                                                    |
                            Static enrichment GRN (with c_h)
                                                    |
                            Interpretable Multi-Head Attention
                                                    |
                            GateAddNorm(attn_out, enriched)
                                                    |
                            Position-wise GRN
                                                    |
                            GateAddNorm(ff_out, attn_enriched)
                                                    |
                            Linear(last_timestep) --> [B, out_size]
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int = 0,
                 hidden_size: int = 256,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 concat_static: bool = False,
                 no_static: bool = False,
                 initial_forget_bias: int = 5,
                 pred_days: int = 0,
                 doy_indices: tuple = (5, 6),
                 doy_std: float = 0.7071,
                 no_attention: bool = False,
                 no_feature_selection: bool = False):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.concat_static = concat_static
        self.no_static = no_static
        self.no_attention = no_attention
        self.no_feature_selection = no_feature_selection
        self.out_size = max(pred_days, 1)

        # Whether we have separate static features to process
        self.has_static = (not no_static and not concat_static
                           and input_size_stat > 0)

        # --- Temporal Variable Selection ---
        self.temporal_vsn = VariableSelectionNetwork(
            n_vars=input_size_dyn,
            d_model=hidden_size,
            dropout=dropout,
            context_size=hidden_size if self.has_static else None
        )
        # Simple linear fallback when VSN is ablated
        if no_feature_selection:
            self.temporal_linear = nn.Linear(input_size_dyn, hidden_size)

        # --- Static Processing (when static features are separate) ---
        if self.has_static:
            self.static_vsn = VariableSelectionNetwork(
                n_vars=input_size_stat,
                d_model=hidden_size,
                dropout=dropout
            )
            # 4 static covariate encoders
            self.ctx_vsn = GRN(hidden_size, hidden_size, hidden_size, dropout)
            self.ctx_h0 = GRN(hidden_size, hidden_size, hidden_size, dropout)
            self.ctx_c0 = GRN(hidden_size, hidden_size, hidden_size, dropout)
            self.ctx_enr = GRN(hidden_size, hidden_size, hidden_size, dropout)

            # Static enrichment GRN (conditions temporal features on c_h)
            self.static_enrichment = GRN(
                hidden_size, hidden_size, hidden_size, dropout,
                context_size=hidden_size
            )

        # --- LSTM Encoder ---
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )
        self._init_lstm(initial_forget_bias)
        self.lstm_gate = GateAddNorm(hidden_size, dropout)

        # --- Self-Attention ---
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, n_heads, dropout
        )
        self.attention_gate = GateAddNorm(hidden_size, dropout)

        # --- Position-wise Feed Forward ---
        self.positionwise = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.output_gate = GateAddNorm(hidden_size, dropout)

        # --- Output Projection ---
        self.output_fc = nn.Linear(hidden_size, self.out_size)

    def _init_lstm(self, forget_bias: int):
        """Initialize LSTM with orthogonal weights and forget gate bias."""
        for name, p in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(p.data)
            elif 'weight_hh' in name:
                n = self.hidden_size
                p.data.zero_()
                for i in range(4):
                    p.data[i * n:(i + 1) * n].copy_(torch.eye(n))
            elif 'bias_ih' in name:
                nn.init.constant_(p.data, 0)
                n = self.hidden_size
                p.data[n:2 * n] = forget_bias
            elif 'bias_hh' in name:
                nn.init.constant_(p.data, 0)

    def forward(self, x_d: torch.Tensor,
                x_s: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through TFT.

        Parameters
        ----------
        x_d : torch.Tensor
            Dynamic input, shape [batch, seq_len, input_size_dyn].
        x_s : torch.Tensor, optional
            Static input, shape [batch, input_size_stat].

        Returns
        -------
        output : torch.Tensor
            Prediction, shape [batch, out_size].
        lstm_output : torch.Tensor
            LSTM hidden states, shape [batch, seq_len, hidden_size].
        attention_weights : torch.Tensor
            Per-head attention weights, shape [batch, n_heads, seq_len, seq_len].
        """
        _, T, _ = x_d.shape

        # ---- Static pathway ----
        c_s = c_h = None
        h_0 = c_0 = None

        if self.has_static and x_s is not None:
            static_selected, _ = self.static_vsn(x_s)       # [B, hidden]
            c_s = self.ctx_vsn(static_selected)              # variable selection context
            c_e = self.ctx_h0(static_selected)               # LSTM h_0 init
            c_c = self.ctx_c0(static_selected)               # LSTM c_0 init
            c_h = self.ctx_enr(static_selected)              # attention enrichment

            h_0 = c_e.unsqueeze(0)  # [1, B, hidden]
            c_0 = c_c.unsqueeze(0)  # [1, B, hidden]

        # ---- Temporal variable selection (with static context) ----
        if self.no_feature_selection:
            temporal_selected = self.temporal_linear(x_d)  # [B, T, hidden]
        else:
            ctx_expanded = (c_s.unsqueeze(1).expand(-1, T, -1)
                            if c_s is not None else None)
            temporal_selected, _ = self.temporal_vsn(x_d, ctx_expanded)  # [B, T, hidden]

        # ---- LSTM encoder ----
        if h_0 is not None:
            lstm_out, _ = self.lstm(temporal_selected, (h_0, c_0))
        else:
            lstm_out, _ = self.lstm(temporal_selected)

        # Gated skip connection: GLU(lstm_out) + temporal_selected
        enriched = self.lstm_gate(lstm_out, temporal_selected)

        # ---- Static enrichment (condition on catchment attributes) ----
        if self.has_static and c_h is not None:
            c_h_expanded = c_h.unsqueeze(1).expand(-1, T, -1)
            enriched = self.static_enrichment(enriched, c_h_expanded)

        # ---- Temporal self-attention ----
        if self.no_attention:
            attn_weights = torch.zeros(1, device=x_d.device)
        else:
            attn_out, attn_weights = self.attention(enriched, enriched, enriched)
            enriched = self.attention_gate(attn_out, enriched)

        # ---- Position-wise feed forward ----
        ff_out = self.positionwise(enriched)
        final = self.output_gate(ff_out, enriched)

        # ---- Output: direct linear from last position (like baselines) ----
        output = self.output_fc(final[:, -1, :])  # [B, out_size]

        return output, lstm_out, attn_weights



class _VanillaGLU(nn.Module):
    """GLU with 'linear' attribute name (matches vanilla TFT checkpoints)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return F.glu(x, dim=-1)


class _VanillaGRN(nn.Module):
    """GRN with vanilla attribute names (linear1/linear2/context_linear/skip_linear)."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.1,
                 context_size: Optional[int] = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, hidden_size)

        if context_size is not None:
            self.context_linear = nn.Linear(context_size, hidden_size, bias=False)

        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = _VanillaGLU(output_size)

        # Skip connection
        if input_size != output_size:
            self.skip_linear = nn.Linear(input_size, output_size)
        else:
            self.skip_linear = None

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.linear1(x)

        if context is not None and self.context_size is not None:
            a = a + self.context_linear(context)

        a = F.elu(a)
        a = self.linear2(a)
        a = self.dropout(a)
        g = self.glu(a)

        if self.skip_linear is not None:
            x = self.skip_linear(x)

        return self.layer_norm(g + x)


class _VanillaMultiHeadAttention(nn.Module):
    """Single-head attention used as a component of _VanillaInterpretableMultiHeadAttention."""

    def __init__(self,
                 d_model: int,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V), attention_weights

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = query.size()

        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        output = self.w_o(attn_output)

        return output, attn_weights


class _VanillaInterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention with per-head value projections.

    Unlike the v3f version which shares V across heads, this creates
    separate single-head attention modules (each with its own Q/K/V/O
    projections) and averages their outputs.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Use separate attention heads for interpretability
        self.attention_heads = nn.ModuleList([
            _VanillaMultiHeadAttention(d_model, 1, dropout) for _ in range(n_heads)
        ])

        self.w_h = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        head_outputs = []
        attention_weights = []

        for head in self.attention_heads:
            head_out, head_weights = head(query, key, value, mask)
            head_outputs.append(head_out)
            attention_weights.append(head_weights)

        # Average across heads for final output
        H = torch.stack(head_outputs, dim=0).mean(dim=0)
        output = self.w_h(H)

        # Stack attention weights
        attention_weights = torch.stack(attention_weights, dim=1)  # [batch, n_heads, seq, seq]

        return output, attention_weights



class VanillaTFT(nn.Module):
    """Simplified TFT for nowcasting (pred_days=0).

    Uses simpler residual connections (add + LayerNorm instead of GateAddNorm),
    GRN-based variable selection (instead of VSN), per-head value projections
    in attention, and the custom LSTM from lstm.py. Better suited for
    nowcasting where the target is the last timestep of the input window.
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int = 0,
                 hidden_size: int = 256,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 concat_static: bool = False,
                 no_static: bool = False,
                 initial_forget_bias: int = 5):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        # Input embedding for dynamic features
        self.input_embedding = nn.Linear(input_size_dyn, hidden_size)

        # Static context network (only used when static features are NOT concatenated)
        if not no_static and not concat_static and input_size_stat > 0:
            self.static_context_grn = _VanillaGRN(
                input_size=input_size_stat,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
        else:
            self.static_context_grn = None

        # Variable selection networks
        self.variable_selection_grn = _VanillaGRN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size if self.static_context_grn else None
        )

        # LSTM for temporal processing (custom LSTM from lstm.py)
        self.lstm = CustomLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            initial_forget_bias=initial_forget_bias
        )

        # Temporal self-attention
        self.self_attention = _VanillaInterpretableMultiHeadAttention(
            d_model=hidden_size,
            n_heads=n_heads,
            dropout=dropout
        )

        # Position-wise feed forward
        self.position_wise_grn = _VanillaGRN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_size, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.lstm_dropout = nn.Dropout(dropout)

    def forward(self, x_d: torch.Tensor, x_s: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VanillaTFT.

        Parameters
        ----------
        x_d : torch.Tensor
            Dynamic input tensor of shape [batch_size, seq_len, input_size_dyn]
        x_s : torch.Tensor, optional
            Static input tensor of shape [batch_size, input_size_stat]

        Returns
        -------
        output : torch.Tensor
            Model output of shape [batch_size, 1]
        lstm_output : torch.Tensor
            LSTM hidden states for compatibility
        attention_weights : torch.Tensor
            Attention weights for interpretability
        """
        batch_size, seq_len, _ = x_d.shape

        # Input embedding
        embedded_input = self.input_embedding(x_d)  # [batch, seq, hidden]

        # Static context processing (if not concatenated and not no_static)
        static_context = None
        if not self.no_static and not self.concat_static and x_s is not None and self.static_context_grn is not None:
            static_context = self.static_context_grn(x_s)  # [batch, hidden]
            # Expand to match sequence length for context
            static_context = static_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Variable selection
        selected_input = self.variable_selection_grn(embedded_input, static_context)

        # LSTM processing
        lstm_output, c_n = self.lstm(selected_input)
        lstm_output = self.lstm_dropout(lstm_output)

        # Self-attention
        attended_output, attention_weights = self.self_attention(
            lstm_output, lstm_output, lstm_output
        )

        # Residual connection
        attended_output = self.layer_norm(attended_output + lstm_output)

        # Position-wise processing
        processed_output = self.position_wise_grn(attended_output)

        # Final output (use last time step)
        final_hidden = processed_output[:, -1, :]  # [batch, hidden]
        output = self.output_projection(final_hidden)  # [batch, 1]

        return output, lstm_output, attention_weights
