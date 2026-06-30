"""5D loss helpers and volume metrics for 3D multicoil fine-tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def complex_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,2,D,H,W] complex tensor to [B,1,D,H,W] magnitude."""
    if x.ndim != 5 or x.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, D, H, W], got {tuple(x.shape)}")
    return torch.sqrt(x[:, 0:1].square() + x[:, 1:2].square())


def current_magnitude_mean(noisy: torch.Tensor) -> torch.Tensor:
    """Per-sample current magnitude mean with shape [B,1,1,1,1]."""
    if noisy.ndim != 5 or noisy.shape[1] < 2:
        raise ValueError(f"Expected [B, C>=2, D, H, W], got {tuple(noisy.shape)}")
    magnitude = torch.sqrt(noisy[:, 0:1].square() + noisy[:, 1:2].square())
    scale = magnitude.mean(dim=(-3, -2, -1), keepdim=True)
    if not torch.isfinite(scale).all():
        raise ValueError("Current-mean loss normalization received non-finite scale")
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def _metric_nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.sum((prediction - target) ** 2))
    denominator = float(np.sum(target**2))
    return numerator / max(denominator, 1.0e-12)


def _metric_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    peak = float(np.max(target))
    return 20.0 * np.log10(max(peak, 1.0e-6)) - 10.0 * np.log10(max(mse, 1.0e-12))


def _metric_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        structural_similarity = None

    if prediction.ndim == 3:
        values = [_metric_ssim(target[index], prediction[index]) for index in range(prediction.shape[0])]
        return float(np.mean(values)) if values else float("nan")
    data_range = float(np.max(target) - np.min(target))
    if data_range <= 0:
        return float("nan")
    if structural_similarity is not None:
        return float(structural_similarity(target, prediction, data_range=data_range))
    pred64 = prediction.astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(pred64.mean())
    mu_y = float(target64.mean())
    var_x = float(pred64.var())
    var_y = float(target64.var())
    cov_xy = float(((pred64 - mu_x) * (target64 - mu_y)).mean())
    return ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )


def _volume_metric_fns():
    try:
        from fastmri.evaluate import nmse, psnr, ssim
    except Exception:
        return {"psnr": _metric_psnr, "ssim": _metric_ssim, "nmse": _metric_nmse}
    return {"psnr": psnr, "ssim": ssim, "nmse": nmse}


def _as_float_metric(value: Any) -> float:
    arr = np.asarray(value)
    return float(arr.mean())


def restore_magnitude_volumes(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Restore per-slice preprocessing scales and return volume magnitudes."""
    pred_mag = complex_magnitude(prediction).detach().cpu().float().numpy()[:, 0]
    target_mag = complex_magnitude(clean).detach().cpu().float().numpy()[:, 0]
    if len(metadata) != pred_mag.shape[0]:
        raise ValueError(f"metadata length {len(metadata)} does not match batch {pred_mag.shape[0]}")

    volumes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, entry in enumerate(metadata):
        scales = np.asarray(entry.get("slice_scales"), dtype=np.float32)
        if scales.ndim != 1 or scales.shape[0] != pred_mag.shape[1]:
            raise ValueError(
                f"slice_scales length must equal D={pred_mag.shape[1]} for "
                f"{entry.get('volume_name', 'unknown_volume')}"
            )
        restored_pred = pred_mag[index] * scales[:, None, None]
        restored_target = target_mag[index] * scales[:, None, None]
        volumes.append(
            (
                str(entry.get("volume_name", "unknown_volume")),
                restored_pred.astype(np.float32, copy=False),
                restored_target.astype(np.float32, copy=False),
            )
        )
    return volumes


def compute_volume_metrics(volumes: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Compute mean PSNR, SSIM, and NMSE across restored volumes."""
    if not volumes:
        return {"psnr": float("nan"), "ssim": float("nan"), "nmse": float("nan")}
    metric_fns = _volume_metric_fns()
    metric_lists: dict[str, list[float]] = {"psnr": [], "ssim": [], "nmse": []}
    for _name, prediction, target in volumes:
        metric_lists["psnr"].append(_as_float_metric(metric_fns["psnr"](target, prediction)))
        metric_lists["ssim"].append(_as_float_metric(metric_fns["ssim"](target, prediction)))
        metric_lists["nmse"].append(_as_float_metric(metric_fns["nmse"](target, prediction)))
    return {name: float(np.mean(values)) for name, values in metric_lists.items()}
