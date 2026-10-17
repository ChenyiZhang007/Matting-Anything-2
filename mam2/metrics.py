"""MatAnyone2/RVM alpha metrics, evaluated in float32 at the original resolution."""

import numpy as np
import torch
from torch.nn import functional as F


class AlphaMetrics:
    """MAD, MSE, Gaussian-gradient error and temporal derivative error (dtSSD)."""

    names = ("pha_mad", "pha_mse", "pha_grad", "pha_dtssd")

    def __init__(self, device="cpu", sigma=1.4):
        if sigma <= 0:
            raise ValueError("Gradient sigma must be positive")
        # Match the reference's float64 kernel construction and L2 normalization.
        half = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * 1e-2)))
        size = int(2 * half + 1)
        kernel = np.zeros((size, size))

        def gaussian(x):
            return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

        for i in range(size):
            for j in range(size):
                x, y = i - half, j - half
                kernel[i, j] = gaussian(x) * (-y * gaussian(y) / sigma**2)
        kernel /= np.sqrt((kernel**2).sum())
        self.kernels = [
            torch.from_numpy(k.copy()).to(device=device, dtype=torch.float32)[None, None]
            for k in (kernel, kernel.T)
        ]
        self.padding = size // 2

    def gradient(self, alpha):
        padded = F.pad(alpha[None, None], [self.padding] * 4, mode="replicate")
        dx, dy = [F.conv2d(padded, kernel)[0, 0] for kernel in self.kernels]
        return (dx**2 + dy**2).sqrt()

    @torch.inference_mode()
    def __call__(self, pred, truth, previous_pred=None, previous_truth=None):
        """Inputs are H×W float32 alpha tensors in [0, 1]; first-frame dtSSD is zero."""
        if pred.dtype != torch.float32 or truth.dtype != torch.float32:
            raise ValueError("Metrics require float32 alpha tensors")
        if pred.ndim != 2 or pred.shape != truth.shape:
            raise ValueError("Prediction and ground truth must have the same H×W shape")
        if (previous_pred is None) != (previous_truth is None):
            raise ValueError("Both previous-frame tensors are required for dtSSD")
        error = pred - truth
        dtssd = pred.new_zeros(())
        if previous_pred is not None:
            if previous_pred.shape != pred.shape or previous_truth.shape != truth.shape:
                raise ValueError("Frame dimensions must remain constant within each clip")
            temporal = ((pred - previous_pred) - (truth - previous_truth)) ** 2
            dtssd = (temporal.sum() / truth.numel()).sqrt() * 1e2
        return dict(
            pha_mad=error.abs().mean() * 1e3,
            pha_mse=(error**2).mean() * 1e3,
            pha_grad=((self.gradient(truth) - self.gradient(pred)) ** 2).sum() / 1e3,
            pha_dtssd=dtssd,
        )
