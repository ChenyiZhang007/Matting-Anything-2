"""Lightweight trimap-based matter backed by MEMatte (paper Appendix A.1)."""

import torch
import torch.nn.functional as F
from torch import nn


class LightweightMatter(nn.Module):
    """Normalize RGB, concatenate its trimap, and decode a full-resolution alpha."""

    def __init__(self, backbone, decoder, pixel_mean, pixel_std, size_divisibility=32):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.size_divisibility = size_divisibility
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1), persistent=False
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, image, trimap, return_routing=False, return_features=False):
        """Accept RGB and trimap tensors in [0, 1], both shaped [B, C, H, W]."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected an RGB tensor with shape [B, 3, H, W]")
        if trimap.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError("Trimap must have shape [B, 1, H, W] matching the RGB input")
        height, width = image.shape[-2:]
        inputs = torch.cat(((image - self.pixel_mean) / self.pixel_std, trimap), dim=1)
        multiple = self.size_divisibility
        # Modulo twice avoids padding a dimension already divisible by the stride.
        inputs = F.pad(inputs, (0, (-width) % multiple, 0, (-height) % multiple))
        features, *routing = self.backbone(inputs)
        alpha = self.decoder(features, inputs)["phas"][..., :height, :width]
        if return_features:
            return alpha, routing[0], features
        return (alpha, routing[0]) if return_routing else alpha
