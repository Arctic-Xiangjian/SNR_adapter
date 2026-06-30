"""3D sliding-window inference for multicoil SNRAware volumes."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import OverlapShape3D, PatchShape3D


def patch_positions(size: int, patch: int, overlap: int) -> list[int]:
    """Return patch starts that cover an axis exactly to the final voxel."""
    size = int(size)
    patch = int(patch)
    overlap = int(overlap)
    if patch > size:
        raise ValueError(f"patch={patch} must fit inside size={size}")
    if overlap < 0 or overlap >= patch:
        raise ValueError(f"overlap={overlap} must be in [0, patch)")
    if patch == size:
        return [0]
    step = max(1, patch - overlap)
    positions = list(range(0, size - patch + 1, step))
    final = size - patch
    if positions[-1] != final:
        positions.append(final)
    return positions


def _ramp_axis(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(int(size), device=device, dtype=dtype)
    center = (float(size) - 1.0) / 2.0
    denom = center + 1.0
    weight = 1.0 - torch.abs(coords - center) / denom
    return (weight / weight.max().clamp_min(1.0e-12)).clamp_min(1.0e-3)


def ramp_weight_3d(
    patch: PatchShape3D,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return separable [1,1,D,H,W] ramp weights."""
    wd = _ramp_axis(patch.depth, device=device, dtype=dtype)
    wh = _ramp_axis(patch.height, device=device, dtype=dtype)
    ww = _ramp_axis(patch.width, device=device, dtype=dtype)
    return (wd[:, None, None] * wh[None, :, None] * ww[None, None, :])[None, None]


@torch.no_grad()
def predict_sliding_window_3d(
    model: nn.Module,
    noisy: torch.Tensor,
    *,
    patch: PatchShape3D,
    overlap: OverlapShape3D,
    patch_batch_size: int,
) -> torch.Tensor:
    """Run 3D patch inference and ramp-blend back to [1,2,D,H,W]."""
    if noisy.ndim != 5 or noisy.shape[0] != 1 or noisy.shape[1] != 3:
        raise ValueError(f"Expected noisy [1, 3, D, H, W], got {tuple(noisy.shape)}")
    patch = PatchShape3D.from_value(patch)
    overlap = OverlapShape3D.from_value(overlap)
    _batch, _channels, depth, height, width = noisy.shape
    z_positions = patch_positions(depth, patch.depth, overlap.depth)
    y_positions = patch_positions(height, patch.height, overlap.height)
    x_positions = patch_positions(width, patch.width, overlap.width)

    prediction_sum = torch.zeros(1, 2, depth, height, width, device=noisy.device, dtype=torch.float32)
    weight_sum = torch.zeros(1, 1, depth, height, width, device=noisy.device, dtype=torch.float32)
    weight = ramp_weight_3d(patch, device=noisy.device, dtype=torch.float32)
    pending: list[torch.Tensor] = []
    coords: list[tuple[int, int, int]] = []
    patch_budget = max(1, int(patch_batch_size))

    def flush() -> None:
        if not pending:
            return
        chunk = torch.cat(pending, dim=0)
        output = model(chunk).float()
        if output.ndim != 5 or output.shape[1:] != (2, patch.depth, patch.height, patch.width):
            raise ValueError(f"Expected model patch output [N,2,D,H,W], got {tuple(output.shape)}")
        for cursor, (z, y, x) in enumerate(coords):
            patch_output = output[cursor : cursor + 1]
            prediction_sum[:, :, z : z + patch.depth, y : y + patch.height, x : x + patch.width] += (
                patch_output * weight
            )
            weight_sum[:, :, z : z + patch.depth, y : y + patch.height, x : x + patch.width] += weight
        pending.clear()
        coords.clear()

    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                pending.append(noisy[:, :, z : z + patch.depth, y : y + patch.height, x : x + patch.width])
                coords.append((z, y, x))
                if len(pending) >= patch_budget:
                    flush()
    flush()
    if not bool((weight_sum > 0).all().item()):
        raise RuntimeError("3D sliding-window produced uncovered voxels")
    return prediction_sum / weight_sum.clamp_min(1.0e-8)
