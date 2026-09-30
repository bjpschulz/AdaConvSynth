import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# -------------------------
# Building blocks
# -------------------------


class DenseLayer(nn.Module):
    """
    A single DenseNet-style layer:
      x -> conv3x3 -> LeakyReLU -> concat([x, new_features])
    """

    def __init__(self, in_ch: int, growth_ch: int, negative_slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth_ch, kernel_size=3, padding=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv(x))
        return torch.cat([x, y], dim=1)


class DenseBlock(nn.Module):
    """
    Stack of DenseLayers. Channel count grows by growth_ch per layer.
    """

    def __init__(
        self, in_ch: int, n_layers: int, growth_ch: int, negative_slope: float = 0.2
    ):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(DenseLayer(ch, growth_ch, negative_slope=negative_slope))
            ch += growth_ch
        self.net = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transition1x1(nn.Module):
    """
    1x1 conv (+ LeakyReLU) to compress/reshape features after dense growth.
    """

    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UpsampleConvT(nn.Module):
    """
    Upsampling via ConvTranspose2d 3x3, stride=2 (diagram-style).
    """

    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.2):
        super().__init__()
        # 3x3, stride=2, padding=1, output_padding=1 doubles spatial size exactly
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.deconv(x))


# -------------------------
# Network
# -------------------------


@dataclass
class SRDenseNetConfig:
    # Required by your prompt (with defaults)
    in_channels: int = 2
    out_channels: int = 2
    init_filters: int = 16
    n_conv_layers: int = (
        8  # interpreted as "number of dense layers per dense block" (see below)
    )
    input_dim: int = 160
    output_dim: int = 320

    # Sensible extras (optional; keep defaults if you do not care)
    n_dense_blocks: int = 8  # matches the second diagram (Dense block 1..8)
    growth_rate: Optional[int] = None  # if None, defaults to init_filters
    bottleneck_channels: int = 256  # post-dense 1x1 compression (diagram shows 256)
    negative_slope: float = 0.2  # LeakyReLU slope
    final_activation: Optional[str] = None  # e.g. "tanh" or None


class SRDenseNet(nn.Module):
    """
    DenseNet-style super-resolution network consistent with the attached diagrams:

      input (C_in, H_in, W_in)
        -> conv3x3 + LeakyReLU (init_filters)
        -> DenseBlock x n_dense_blocks (each with n_conv_layers dense layers, growth growth_rate)
        -> 1x1 conv + LeakyReLU (bottleneck_channels)
        -> upsampling (ConvTranspose2d 3x3 + LeakyReLU) repeated for scale factor
        -> conv3x3 -> out_channels

    Notes:
    - The diagrams show ReLU; this implementation uses LeakyReLU everywhere per your request.
    - Scale factor is inferred from (output_dim / input_dim) and must be an integer power of 2.
      If input_dim == output_dim, upsampling is skipped.
    """

    def __init__(self, cfg: SRDenseNetConfig):
        super().__init__()
        self.cfg = cfg

        growth = cfg.growth_rate if cfg.growth_rate is not None else cfg.init_filters

        # Validate spatial scaling
        if cfg.output_dim % cfg.input_dim != 0:
            raise ValueError(
                f"output_dim ({cfg.output_dim}) must be a multiple of input_dim ({cfg.input_dim})."
            )
        scale = cfg.output_dim // cfg.input_dim
        if scale < 1:
            raise ValueError("output_dim must be >= input_dim.")
        if scale != 1 and (scale & (scale - 1)) != 0:
            raise ValueError(
                f"Scale factor must be a power of 2 for ConvTranspose-based upsampling. Got scale={scale}."
            )
        self.scale = scale

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                cfg.in_channels, cfg.init_filters, kernel_size=3, padding=1, bias=True
            ),
            nn.LeakyReLU(negative_slope=cfg.negative_slope, inplace=True),
        )

        # Dense trunk
        blocks = []
        ch = cfg.init_filters
        for _ in range(cfg.n_dense_blocks):
            db = DenseBlock(
                ch,
                n_layers=cfg.n_conv_layers,
                growth_ch=growth,
                negative_slope=cfg.negative_slope,
            )
            blocks.append(db)
            ch = db.out_ch
        self.dense_trunk = nn.Sequential(*blocks)

        # 1x1 transition (diagram shows conv 1x1 + ReLU producing 256 channels)
        self.transition = Transition1x1(
            ch, cfg.bottleneck_channels, negative_slope=cfg.negative_slope
        )

        # Upsampling path (diagram shows convT 3x3 + ReLU then a final conv 3x3)
        up_layers = []
        up_ch = cfg.bottleneck_channels
        n_up = int(math.log2(scale)) if scale > 1 else 0
        for _ in range(n_up):
            up_layers.append(
                UpsampleConvT(up_ch, up_ch, negative_slope=cfg.negative_slope)
            )
        self.upsampler = nn.Sequential(*up_layers)

        # Head
        self.head = nn.Conv2d(
            up_ch, cfg.out_channels, kernel_size=3, padding=1, bias=True
        )

        if cfg.final_activation is None:
            self.final_act = None
        elif cfg.final_activation.lower() == "tanh":
            self.final_act = nn.Tanh()
        elif cfg.final_activation.lower() == "sigmoid":
            self.final_act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported final_activation: {cfg.final_activation}")

        self.model_name = "srdensenet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x is a 2D complex-valued image
        # -> transform to real, apply nn, transform back
        x = torch.view_as_real(x)
        nb, ny, nx, nc = x.shape
        x = rearrange(x, "b y x c -> b c y x")

        # Optional safety check: enforce expected input size if user relies on cfg.input_dim
        if x.dim() != 4:
            raise ValueError("Expected input of shape (N, C, H, W).")
        if x.size(1) != self.cfg.in_channels:
            raise ValueError(
                f"Expected {self.cfg.in_channels} input channels, got {x.size(1)}."
            )

        y = self.stem(x)
        y = self.dense_trunk(y)
        y = self.transition(y)
        y = self.upsampler(y)
        y = self.head(y)
        if self.final_act is not None:
            y = self.final_act(y)

        # bring back to complex-valued repr
        y = rearrange(y, "b c y x -> b y x c", b=nb, c=nc)
        y = torch.view_as_complex(y.contiguous())
        return y


# # -------------------------
# # Minimal usage / shape test
# # -------------------------
# if __name__ == "__main__":


# cfg = SRDenseNetConfig(
#     in_channels=2,
#     out_channels=2,
#     init_filters=16,
#     n_conv_layers=8,
#     input_dim=64,
#     output_dim=128,  # scale = 2
#     n_dense_blocks=8,
#     bottleneck_channels=256,
#     negative_slope=0.2,
# )
# net = SRDenseNet(cfg)

# x = torch.randn(4, cfg.in_channels, cfg.input_dim, cfg.input_dim)
# y = net(x)
# print("Output shape:", tuple(y.shape))  # (4, 2, 128, 128)
