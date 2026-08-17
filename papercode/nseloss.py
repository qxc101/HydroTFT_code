"""HydroTFT -- basin-normalised NSE loss with optional lead-time weighting.

Modified from the file of the same name in
    https://github.com/kratzert/ealstm_regional_modeling  (Apache-2.0)
for the HydroTFT experiments. Modifications are listed in the accompanying README.
Distributed under the Apache License 2.0; see the LICENSE file.
"""

import torch


class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1, horizon_alpha: float = 0.0):
        super(NSELoss, self).__init__()
        self.eps = eps
        self.horizon_alpha = horizon_alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true)**2
        weights = 1 / (q_stds + self.eps)**2
        scaled_loss = weights * squared_error

        # Horizon weighting: linearly increase weight for later forecast days
        if self.horizon_alpha > 0 and y_pred.dim() >= 2 and y_pred.shape[-1] > 1:
            T = y_pred.shape[-1]
            hw = 1.0 + self.horizon_alpha * torch.arange(T, device=y_pred.device).float() / (T - 1)
            scaled_loss = scaled_loss * hw.unsqueeze(0)

        return torch.mean(scaled_loss)
