"""PatchTST baseline for cross-basin rainfall-runoff prediction.

Implementation of PatchTST of

    Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J.
    "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
    ICLR 2023. https://arxiv.org/abs/2211.14730

adapted to the HydroTFT experimental setup so it consumes exactly the same
inputs as TFT/EA-LSTM/LSTM (5 meteorological forcings + 5 engineered "starter"
features as dynamic channels, and 27 static catchment attributes) and produces
`pred_days` streamflow values.

Design notes (documented for the paper's reproducibility section):
- The backbone is faithful to PatchTST: channel-independent patching (each of the
  C dynamic channels is split into overlapping patches of length `patch_len` with
  the given `stride`, end-padded by `stride` as in the paper's `padding_patch='end'`),
  a shared Linear patch embedding + learnable positional embedding, and a standard
  Transformer encoder shared across channels.
- Because streamflow is deliberately NOT an input channel (rainfall-runoff, not
  autoregression), the per-channel "flatten head" of PatchTST is replaced by a
  joint flatten head that concatenates all channels' patch representations and maps
  them to the `pred_days` streamflow targets.
- Static catchment attributes are fused via a separate linear encoder concatenated
  before the head, mirroring HydroTFT (fair-comparison choice).
- Variants for the reviewer-requested fairness study (defaults reproduce the paper's
  configuration exactly):
    head_mode='joint'    : joint flatten head over all channels (+ statics)   [paper]
    head_mode='channel'  : PatchTST's own per-channel flatten head, weights shared
                           across channels, outputs averaged over channels. This is
                           the closest faithful reading of the original head for a
                           target that is not itself an input channel.
    channels=[0]         : restrict the dynamic input to a subset of channels
                           (channel 0 = precipitation), i.e. a strictly univariate
                           application of PatchTST.
- Architecture hyperparameters use the paper's supervised default: patch_len=16,
  stride=8, d_model=128, n_heads=16, e_layers=3, d_ff=256, GELU, dropout from run
  config. RevIN is omitted because inputs are already standardized per basin and the
  regression target is a separate variable (so PatchTST's denormalization does not
  apply).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Channel-independent patch Transformer adapted for rainfall-runoff regression."""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 pred_days: int,
                 seq_length: int = 365,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 16,
                 e_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.2,
                 head_mode: str = 'joint',
                 channels=None,
                 **kwargs):
        super().__init__()
        assert head_mode in ('joint', 'channel'), head_mode
        self.head_mode = head_mode
        self.channels = list(channels) if channels is not None else None
        self.out_size = max(pred_days, 1)
        # if a channel subset is requested the backbone only ever sees those channels
        self.n_vars = len(self.channels) if self.channels is not None else input_size_dyn
        self.input_size_stat = input_size_stat
        self.patch_len = patch_len
        self.stride = stride
        self.pad = stride  # padding_patch='end' pads `stride` steps at the end
        self.d_model = d_model

        num_patches = int((seq_length + self.pad - patch_len) / stride) + 1
        self.num_patches = num_patches

        # Shared patch embedding (channel independence) + learnable positions.
        self.W_P = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(num_patches, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        if input_size_stat > 0:
            self.static_enc = nn.Sequential(
                nn.Linear(input_size_stat, d_model), nn.GELU())
        else:
            self.static_enc = None

        self.head_dropout = nn.Dropout(dropout)
        if self.head_mode == 'joint':
            head_in = self.n_vars * num_patches * d_model
            if self.static_enc is not None:
                head_in += d_model
            self.head = nn.Linear(head_in, self.out_size)
        else:
            # PatchTST's original per-channel flatten head (shared across channels), applied to
            # each channel's patch sequence and averaged over channels. Statics, if any, are
            # added as a bias term produced by the static encoder.
            self.head = nn.Linear(num_patches * d_model, self.out_size)
            self.static_head = (nn.Linear(d_model, self.out_size)
                                if self.static_enc is not None else None)

        print(f"-> PatchTST: patch_len={patch_len}, stride={stride}, num_patches={num_patches}, "
              f"d_model={d_model}, heads={n_heads}, layers={e_layers}, d_ff={d_ff}, "
              f"dropout={dropout}, n_vars={self.n_vars}, static={input_size_stat}, "
              f"out_size={self.out_size}, seq_length={seq_length}, head_mode={self.head_mode}, "
              f"channels={self.channels}")

    def forward(self, x_d: torch.Tensor, x_s: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.channels is not None:                             # univariate / subset variant
            x_d = x_d[:, :, self.channels]
        # x_d: (B, L, C) -> channel-major (B, C, L)
        B, L, C = x_d.shape
        x = x_d.permute(0, 2, 1)                                   # (B, C, L)
        if self.pad > 0:                                          # end padding by replication
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, self.pad)], dim=-1)
        # Unfold into overlapping patches -> (B, C, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        npatch = x.shape[2]
        x = x.reshape(B * C, npatch, self.patch_len)             # channel-independent
        x = self.W_P(x) + self.pos[:npatch]                      # (B*C, npatch, d_model)
        x = self.embed_dropout(x)
        z = self.encoder(x)                                      # patch attention within channel
        z = self.enc_norm(z)                                     # (B*C, npatch, d_model)

        if self.head_mode == 'joint':
            z = z.reshape(B, C, npatch, self.d_model).reshape(B, -1)  # joint flatten
            if self.static_enc is not None and x_s is not None:
                s = self.static_enc(x_s)                        # (B, d_model)
                z = torch.cat([z, s], dim=-1)
            out = self.head(self.head_dropout(z))               # (B, out_size)
        else:
            z = z.reshape(B * C, npatch * self.d_model)         # per-channel flatten
            out = self.head(self.head_dropout(z))               # (B*C, out_size)
            out = out.reshape(B, C, self.out_size).mean(dim=1)  # average the channel heads
            if self.static_head is not None and x_s is not None:
                out = out + self.static_head(self.static_enc(x_s))
        # Return a 3-tuple to match the TFT interface expected by the Model wrapper.
        return out, None, None
