"""Correction adapter and minimal LoRA support for multicoil fine-tuning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from snraware.components.model import Conv2DExt, Conv3DExt, LinearGrid3DExt, LinearGridExt

from .config import CorrectionConfig, LoraConfig


class PhysicsCorrectionAdapter(nn.Module):
    """Bounded correction on native [real, imag, ones-gmap] inputs."""

    def __init__(
        self,
        config: CorrectionConfig,
        *,
        in_chans: int = 3,
    ):
        super().__init__()
        if in_chans != 3:
            raise ValueError("PhysicsCorrectionAdapter expects [real, imag, gmap]")
        self.gmap_log_bound = float(config.gmap_log_bound)
        self.complex_log_scale_bound = float(config.complex_log_scale_bound)
        self.gmap_min = float(config.gmap_min)
        self.gmap_max = float(config.gmap_max)
        hidden_chans = int(config.hidden_chans)
        self.log_complex_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gmap_net = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_chans, hidden_chans, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_chans, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gmap_net[-1].weight)
        nn.init.zeros_(self.gmap_net[-1].bias)
        self.last_stats: dict[str, float] | None = None

    @staticmethod
    def _prepare_native_input(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 4 and x.shape[1] == 3:
            return x, False
        if x.ndim == 5 and x.shape[1] == 3:
            if x.shape[2] != 1:
                raise ValueError(f"Expected singleton T=1, got {tuple(x.shape)}")
            return x.squeeze(2), True
        raise ValueError(f"Expected [B, 3, H, W] or [B, 3, 1, H, W], got {tuple(x.shape)}")

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
        x_2d, had_time = self._prepare_native_input(x)
        log_scale = self.complex_log_scale_bound * torch.tanh(self.log_complex_scale)
        complex_scale = torch.exp(log_scale).to(device=x_2d.device, dtype=x_2d.dtype)
        gmap_delta = self.gmap_log_bound * torch.tanh(self.gmap_net(x_2d).to(dtype=x_2d.dtype))
        gmap_ratio = torch.exp(gmap_delta)
        corrected_complex = x_2d[:, 0:2] * complex_scale
        corrected_gmap = torch.clamp(x_2d[:, 2:3] * gmap_ratio, self.gmap_min, self.gmap_max)
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
        return out.unsqueeze(2) if had_time else out


class LoRALinear(nn.Module):
    """Low-rank update around a frozen Linear module."""

    def __init__(self, base: nn.Linear, config: LoraConfig):
        super().__init__()
        self.base_layer = base
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        rank = int(config.r)
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = float(config.alpha) / float(rank)
        self.dropout = nn.Dropout(float(config.dropout)) if config.dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    """Low-rank update around a frozen Conv2d module."""

    def __init__(self, base: nn.Conv2d, config: LoraConfig):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv2d only supports groups=1")
        self.base_layer = base
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        rank = int(config.r)
        self.lora_A = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_B = nn.Conv2d(
            rank,
            base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.scaling = float(config.alpha) / float(rank)
        self.dropout = nn.Dropout(float(config.dropout)) if config.dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAConv3d(nn.Module):
    """Low-rank update around a frozen Conv3d module."""

    def __init__(self, base: nn.Conv3d, config: LoraConfig):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv3d only supports groups=1")
        self.base_layer = base
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        rank = int(config.r)
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.lora_A = nn.Conv3d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_B = nn.Conv3d(
            rank,
            base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.scaling = float(config.alpha) / float(rank)
        self.dropout = nn.Dropout(float(config.dropout)) if config.dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


@dataclass
class LoraApplyResult:
    """Summary of injected LoRA modules."""

    num_wrapped: int
    wrapped_names: list[str]


def _matches_target(name: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def _get_parent_module(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def has_lora_adapters(model: nn.Module) -> bool:
    """Return True when model already contains LoRA wrappers."""
    return any(isinstance(module, (LoRALinear, LoRAConv2d, LoRAConv3d)) for module in model.modules())


def _inject_lora_into_extension(module: nn.Module, config: LoraConfig) -> bool:
    if isinstance(module, Conv2DExt):
        if not isinstance(module.conv, LoRAConv2d):
            module.conv = LoRAConv2d(module.conv, config)
        return True
    if isinstance(module, Conv3DExt):
        if not isinstance(module.conv, LoRAConv3d):
            module.conv = LoRAConv3d(module.conv, config)
        return True
    if isinstance(module, (LinearGridExt, LinearGrid3DExt)):
        if not isinstance(module.linear, LoRALinear):
            module.linear = LoRALinear(module.linear, config)
        return True
    return False


def apply_lora_to_model(model: nn.Module, config: LoraConfig) -> LoraApplyResult:
    """Inject LoRA adapters into modules selected by regex patterns."""
    if not config.enabled:
        return LoraApplyResult(num_wrapped=0, wrapped_names=[])
    if has_lora_adapters(model):
        names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, (LoRALinear, LoRAConv2d, LoRAConv3d))
        ]
        return LoraApplyResult(num_wrapped=len(names), wrapped_names=names)

    replacements: list[tuple[str, str, nn.Module]] = []
    injected_extensions: list[str] = []
    for name, module in model.named_modules():
        if not name or not _matches_target(name, config.target_modules):
            continue
        if _inject_lora_into_extension(module, config):
            injected_extensions.append(name)
        elif isinstance(module, nn.Linear):
            replacements.append((name, "linear", LoRALinear(module, config)))
        elif isinstance(module, nn.Conv2d):
            replacements.append((name, "conv2d", LoRAConv2d(module, config)))
        elif isinstance(module, nn.Conv3d):
            replacements.append((name, "conv3d", LoRAConv3d(module, config)))

    wrapped: list[str] = list(injected_extensions)
    for dotted_name, _kind, replacement in replacements:
        parent, child_name = _get_parent_module(model, dotted_name)
        setattr(parent, child_name, replacement)
        wrapped.append(dotted_name)
    if not wrapped:
        raise RuntimeError(
            "LoRA is enabled but no modules matched target_modules. "
            f"Patterns: {config.target_modules}"
        )
    return LoraApplyResult(num_wrapped=len(wrapped), wrapped_names=wrapped)


def set_lora_trainable(model: nn.Module, flag: bool) -> None:
    """Enable or disable gradients for LoRA parameters only."""
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = bool(flag)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable LoRA parameters."""
    return [parameter for name, parameter in model.named_parameters() if "lora_" in name]


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
