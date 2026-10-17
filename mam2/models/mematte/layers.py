# Copyright (c) Facebook, Inc. and its affiliates.
"""Shared PyTorch residual propagation block for the MEMatte student and teacher."""

from collections.abc import Callable

from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d


def _normalization(norm: str | Callable[[int], nn.Module], channels: int) -> nn.Module:
    if callable(norm):
        return norm(channels)
    factories = {
        "LN": LayerNorm2d,
        "BN": nn.BatchNorm2d,
        "SyncBN": nn.SyncBatchNorm,
        "GN": lambda count: nn.GroupNorm(32, count),
    }
    if norm not in factories:
        raise ValueError(
            f"Unsupported normalization {norm!r}; use {tuple(factories)} or a callable"
        )
    return factories[norm](channels)


class ResBottleneckBlock(nn.Module):
    """Three convolutions with channel normalization and an identity residual.

    Parameter names, normalization epsilon and initialization match the historical
    MEMatte blocks, so student and teacher checkpoints retain their original keys.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
        conv_kernels=3,
        conv_paddings=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 1
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = _normalization(norm, bottleneck_channels)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            conv_kernels,
            padding=conv_paddings,
            bias=False,
        )
        self.norm2 = _normalization(norm, bottleneck_channels)
        self.act2 = act_layer()
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = _normalization(norm, out_channels)
        for layer in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        for layer in (self.norm1, self.norm2):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.norm3.weight)
        nn.init.zeros_(self.norm3.bias)

    def forward(self, x):
        residual = x
        for layer in self.children():
            x = layer(x)
        return residual + x
