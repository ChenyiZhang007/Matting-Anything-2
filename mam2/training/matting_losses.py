"""Alpha losses for Stage 2 (paper Appendix A.5)."""

import torch
import torch.nn.functional as F
from torch import nn


class MattingLoss(nn.Module):
    """Separate known/unknown L1, L2, Laplacian and gradient supervision."""

    def __init__(self, weights, laplacian_levels=5, gradient_regularization=0.01):
        super().__init__()
        self.weights = weights
        self.laplacian_levels = laplacian_levels
        self.gradient_regularization = gradient_regularization

    @staticmethod
    def masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1)

    def forward(self, prediction, target, trimap):
        unknown = ((trimap > 0) & (trimap < 1)).to(prediction.dtype)
        known = 1 - unknown
        difference = (prediction - target).abs()
        losses = {
            "unknown_l1": self.masked_mean(difference, unknown),
            "known_l1": self.masked_mean(difference, known),
            "l2": F.mse_loss(prediction, target),
            "laplacian": laplacian_loss(prediction, target, self.laplacian_levels),
        }
        kernel = prediction.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
        gradient = prediction.new_zeros(())
        for sobel in (kernel, kernel.transpose(-1, -2)):
            pred_gradient = F.conv2d(prediction, sobel, padding=1)
            true_gradient = F.conv2d(target, sobel, padding=1)
            gradient = gradient + self.masked_mean((pred_gradient - true_gradient).abs(), unknown)
            gradient = gradient + self.gradient_regularization * self.masked_mean(
                pred_gradient.abs(), unknown
            )
        losses["gradient"] = gradient
        return {name: value * self.weights.get(name, 0) for name, value in losses.items()}


def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2**level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels


def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid


def gauss_kernel(device="cpu", dtype=torch.float32):
    kernel = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        device=device,
        dtype=dtype,
    )
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel


def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode="reflect")
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img


def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img


def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out


def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]
