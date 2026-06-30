"""3D g-factor correction adapter for multicoil SNRAware inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CorrectionConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock3D(nn.Module):
    """Conv3d, GroupNorm, and SiLU block."""

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_chans), out_chans),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_chans), out_chans),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GFactorUNet3D(nn.Module):
    """Input [B,3,D,H,W], output [B,1,D,H,W] log-gmap delta."""

    def __init__(self, in_chans: int = 3, hidden_chans: int = 32):
        super().__init__()
        hidden = int(hidden_chans)
        if hidden <= 0:
            raise ValueError("hidden_chans must be positive")
        self.enc1 = ConvBlock3D(in_chans, hidden)
        self.down1 = nn.Conv3d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock3D(hidden * 2, hidden * 2)
        self.down2 = nn.Conv3d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ConvBlock3D(hidden * 4, hidden * 4)
        self.up2 = ConvBlock3D(hidden * 6, hidden * 2)
        self.up1 = ConvBlock3D(hidden * 3, hidden)
        self.out = nn.Conv3d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, D, H, W], got {tuple(x.shape)}")
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        y = F.interpolate(x3, size=x2.shape[-3:], mode="trilinear", align_corners=False)
        y = self.up2(torch.cat([y, x2], dim=1))
        y = F.interpolate(y, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        y = self.up1(torch.cat([y, x1], dim=1))
        return self.out(y)


class GFactorCorrectionAdapter3D(nn.Module):
    """Bounded 3D correction on native [real, imag, gmap] inputs."""

    def __init__(self, config: CorrectionConfig, *, in_chans: int = 3):
        super().__init__()
        if in_chans != 3:
            raise ValueError("GFactorCorrectionAdapter3D expects [real, imag, gmap]")
        self.gmap_log_bound = float(config.gmap_log_bound)
        self.complex_log_scale_bound = float(config.complex_log_scale_bound)
        self.gmap_min = float(config.gmap_min)
        self.gmap_max = float(config.gmap_max)
        self.log_complex_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gmap_unet = GFactorUNet3D(in_chans=in_chans, hidden_chans=int(config.hidden_chans))
        self.last_stats: dict[str, float] | None = None

    @staticmethod
    def _finite_stats(value: torch.Tensor) -> dict[str, float]:
        finite = value.detach().float().reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {
            "mean": float(finite.mean().item()),
            "p95": float(torch.quantile(finite, 0.95).item()),
            "max": float(finite.max().item()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, D, H, W], got {tuple(x.shape)}")
        log_scale = self.complex_log_scale_bound * torch.tanh(self.log_complex_scale)
        complex_scale = torch.exp(log_scale).to(device=x.device, dtype=x.dtype)
        log_gmap_delta = self.gmap_log_bound * torch.tanh(self.gmap_unet(x).to(dtype=x.dtype))
        gmap_ratio = torch.exp(log_gmap_delta)
        corrected_complex = x[:, 0:2] * complex_scale
        corrected_gmap = torch.clamp(x[:, 2:3] * gmap_ratio, self.gmap_min, self.gmap_max)
        out = torch.cat([corrected_complex, corrected_gmap], dim=1)

        ratio_stats = self._finite_stats(gmap_ratio)
        gmap_stats = self._finite_stats(corrected_gmap)
        self.last_stats = {
            "complex_scale": float(complex_scale.detach().float().item()),
            "ratio_mean": ratio_stats["mean"],
            "ratio_p95": ratio_stats["p95"],
            "ratio_max": ratio_stats["max"],
            "gmap_mean": gmap_stats["mean"],
            "gmap_p95": gmap_stats["p95"],
            "gmap_max": gmap_stats["max"],
        }
        return out
