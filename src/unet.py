"""Conditional pixel-space flow U-Net, trained without a latent autoencoder.

Adaptive GroupNorm residual blocks, low-resolution QK-normalized attention,
lossless 2x2 input packing, and a full-resolution detail skip. The output is
velocity, so the existing flow objective and Euler/Heun samplers apply directly.
"""

import math
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, max(1, channels // 4), channels), channels, eps=1e-5)


class TimeEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width * 4))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(self.width // 2, device=time.device).float()
            / max(1, self.width // 2 - 1)
        )
        angles = time.float()[:, None] * 1000 * frequencies[None]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, time_width: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = group_norm(inputs), group_norm(outputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.affine = nn.Linear(time_width, 2 * outputs)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, shift = self.affine(F.silu(time)).to(hidden.dtype)[:, :, None, None].chunk(2, dim=1)
        hidden = F.silu(self.norm2(hidden) * (1 + scale) + shift)
        return (self.skip(image) + self.conv2(self.dropout(hidden))) * (2 ** -0.5)


class SpatialAttention(nn.Module):
    """Attention only at the bottleneck; full-resolution maps remain convolutional."""

    def __init__(self, channels: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.head_dim = channels // head_dim, head_dim
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        q, k, v = self.qkv(self.norm(image)).reshape(
            batch, 3, self.heads, self.head_dim, height * width
        ).unbind(1)
        q, k, v = q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(-1, -2).reshape(batch, channels, height, width)
        return (image + self.proj(attended)) * (2 ** -0.5)


class Stage(nn.Module):
    def __init__(self, inputs: int, outputs: int, time_width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(inputs if index == 0 else outputs, outputs, time_width, dropout)
            for index in range(blocks)
        ])

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            image = block(image, time)
        return image


class FlowUNet(nn.Module):
    """Depth-conditioned velocity predictor with per-scale spatial conditioning.

    ``feature_layer=1`` exposes bottleneck tokens for optional training-only REPA.
    Inputs are noisy RGB and already encoded continuous near-bright BREA depth.
    Internal pooling builds features; it does not recalibrate depth encoding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        condition_channels: int = 1,
        base_channels: int = 160,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        blocks_per_level: int = 2,
        attention_head_dim: int = 64,
        detail_channels: int = 32,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or condition_channels <= 0:
            raise ValueError("image and condition channels must be positive")
        if base_channels < 4 or base_channels % 4 or not channel_mult or channel_mult[0] != 1:
            raise ValueError("base_channels must be a positive multiple of 4 and channel_mult must start at 1")
        if any(mult <= 0 or int(mult) != mult for mult in channel_mult):
            raise ValueError("channel multipliers must be positive integers")
        if blocks_per_level <= 0 or detail_channels <= 0 or not 0 <= dropout < 1:
            raise ValueError("blocks/detail channels must be positive and dropout must be in [0, 1)")
        channels = [base_channels * mult for mult in channel_mult]
        if attention_head_dim <= 0 or channels[-1] % attention_head_dim:
            raise ValueError("bottleneck channels must be divisible by attention_head_dim")
        self.in_channels, self.condition_channels = in_channels, condition_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.spatial_multiple = 2 ** len(channels)
        self.hidden_size = channels[-1]
        self.patch_size = self.spatial_multiple  # REPA feature stride, not a latent codec.
        self.time_embed = TimeEmbedding(base_channels)
        conditioned = in_channels + condition_channels
        self.stem = nn.Conv2d(conditioned * 4, base_channels, 3, padding=1)
        self.detail = nn.Conv2d(conditioned, detail_channels, 3, padding=1)
        self.condition_projections = nn.ModuleList([
            nn.Conv2d(condition_channels, width, 1) for width in channels
        ])
        self.encoder = nn.ModuleList([
            Stage(width, width, base_channels * 4, blocks_per_level, dropout) for width in channels
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(left, right, 3, padding=1) for left, right in pairwise(channels)
        ])
        self.middle1 = ResidualBlock(channels[-1], channels[-1], base_channels * 4, dropout)
        self.attention = SpatialAttention(channels[-1], attention_head_dim)
        self.middle2 = ResidualBlock(channels[-1], channels[-1], base_channels * 4, dropout)
        self.decoder = nn.ModuleList([
            Stage(width * 2, width, base_channels * 4, blocks_per_level, dropout)
            for width in reversed(channels)
        ])
        self.upsample = nn.ModuleList([
            nn.Conv2d(channels[index], channels[index - 1], 3, padding=1)
            for index in range(len(channels) - 1, 0, -1)
        ])
        head_channels = base_channels // 4 + detail_channels
        self.output_norm = group_norm(head_channels)
        self.output_conv = nn.Conv2d(head_channels, detail_channels, 3, padding=1)
        self.final = nn.Conv2d(detail_channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def _run(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, depth: torch.Tensor,
        feature_layer: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("expected NCHW image with configured in_channels")
        batch, _, height, width = x.shape
        if height % self.spatial_multiple or width % self.spatial_multiple:
            raise ValueError(f"image dimensions must be divisible by {self.spatial_multiple}")
        if depth.shape != (batch, self.condition_channels, height, width):
            raise ValueError("depth must match the image batch, spatial size, and condition_channels")
        if t.numel() != batch:
            raise ValueError("expected one timestep per image")
        if feature_layer is not None and feature_layer != 1:
            raise ValueError("FlowUNet exposes bottleneck REPA features at feature_layer=1")
        time = self.time_embed(t.reshape(batch))
        depth = depth.to(device=x.device, dtype=x.dtype)
        conditioned = torch.cat((x, depth), dim=1)
        detail = F.silu(self.detail(conditioned))
        hidden = self.stem(F.pixel_unshuffle(conditioned, 2))
        skips = []
        condition = F.avg_pool2d(depth, 2)
        for index, stage in enumerate(self.encoder):
            hidden = hidden + self.condition_projections[index](condition)
            hidden = self._run(stage, hidden, time)
            skips.append(hidden)
            if index < len(self.downsample):
                hidden = self.downsample[index](F.avg_pool2d(hidden, 2))
                condition = F.avg_pool2d(condition, 2)
        hidden = self._run(self.middle1, hidden, time)
        hidden = self._run(self.attention, hidden)
        hidden = self._run(self.middle2, hidden, time)
        features = hidden.flatten(2).transpose(1, 2) if feature_layer is not None else None
        for index, stage in enumerate(self.decoder):
            hidden = self._run(stage, torch.cat((hidden, skips.pop()), dim=1), time)
            if index < len(self.upsample):
                hidden = self.upsample[index](F.interpolate(hidden, scale_factor=2, mode="nearest"))
        hidden = torch.cat((F.pixel_shuffle(hidden, 2), detail), dim=1)
        hidden = F.silu(self.output_conv(F.silu(self.output_norm(hidden))))
        velocity = self.final(hidden)
        return (velocity, features) if feature_layer is not None else velocity
