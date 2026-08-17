"""iTransformer baseline for cross-basin rainfall-runoff prediction.

Implementation of the Inverted Transformer (iTransformer) of

    Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., Long, M.
    "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting."
    ICLR 2024. https://arxiv.org/abs/2310.06625

adapted to the HydroTFT experimental setup so it consumes exactly the same
inputs as TFT/EA-LSTM/LSTM (5 meteorological forcings + 5 engineered "starter"
features as dynamic channels, and 27 static catchment attributes) and produces
`pred_days` streamflow values.

Design notes (documented for the paper's reproducibility section):
- The backbone is faithful to iTransformer: each dynamic *variate* (channel) has
  its whole `seq_length`-day series embedded into a single token via a shared
  Linear(seq_length -> d_model); a standard Transformer encoder then applies
  attention *across variate tokens*.
- Because streamflow is deliberately NOT an input channel (this is rainfall-runoff,
  not autoregression), we cannot read the target off an output channel. We
  therefore replace iTransformer's per-variate projection with a regression head
  that maps the encoded variate tokens to the `pred_days` streamflow targets.
- Static catchment attributes are fused via a separate linear encoder whose output
  is concatenated with the flattened variate representation before the head,
  exactly mirroring how HydroTFT conditions on statics (fair-comparison choice).
- Architecture hyperparameters use the paper's defaults: d_model=512, e_layers=3,
  n_heads=8, d_ff=512, GELU, dropout from the run config.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class iTransformer(nn.Module):
    """Inverted Transformer adapted for rainfall-runoff regression."""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 pred_days: int,
                 seq_length: int = 365,
                 d_model: int = 512,
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.out_size = max(pred_days, 1)
        self.n_vars = input_size_dyn
        self.input_size_stat = input_size_stat
        self.seq_length = seq_length
        self.d_model = d_model

        # Variate embedding: entire per-channel series (length L) -> one token.
        self.embed = nn.Linear(seq_length, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        head_in = self.n_vars * d_model
        if input_size_stat > 0:
            self.static_enc = nn.Sequential(
                nn.Linear(input_size_stat, d_model), nn.GELU())
            head_in += d_model
        else:
            self.static_enc = None

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(head_in, self.out_size)

        print(f"-> iTransformer: d_model={d_model}, heads={n_heads}, layers={e_layers}, "
              f"d_ff={d_ff}, dropout={dropout}, n_vars={self.n_vars}, "
              f"static={input_size_stat}, out_size={self.out_size}, seq_length={seq_length}")

    def forward(self, x_d: torch.Tensor, x_s: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x_d: (B, L, C) -> invert to variate-major (B, C, L)
        x = x_d.permute(0, 2, 1)
        tokens = self.embed(x)               # (B, C, d_model): one token per variate
        tokens = self.embed_dropout(tokens)
        z = self.encoder(tokens)             # attention across variate tokens
        z = self.enc_norm(z)                 # (B, C, d_model)
        z = z.reshape(z.shape[0], -1)        # (B, C * d_model)

        if self.static_enc is not None and x_s is not None:
            s = self.static_enc(x_s)         # (B, d_model)
            z = torch.cat([z, s], dim=-1)

        out = self.head(self.head_dropout(z))  # (B, out_size)
        # Return a 3-tuple to match the TFT interface expected by the Model wrapper.
        return out, None, None
