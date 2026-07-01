"""Pure multicoil physics path with ones-gmap plus learned correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PreprocessConfig
from .preprocess import (
    EPS,
    apply_coil_matrix,
    build_1d_mask,
    center_crop_or_pad_real,
    check_complex_coil_kspace,
    estimate_csm,
    estimate_noise_covariance,
    estimate_scc_matrix,
    extract_center_calib,
    ifft2c_np,
    seed_from_name,
    sense_combine,
    to_uniform_kspace,
)

try:
    from pygrappa import grappa as pygrappa_grappa
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    pygrappa_grappa = None


@dataclass
class MulticoilPreprocessResult:
    """Native SNRAware arrays for one multicoil slice."""

    noisy_complex: np.ndarray
    clean_complex: np.ndarray
    gmap: np.ndarray
    metadata: dict[str, Any]
    zero_filled_complex: np.ndarray | None = None
    target_rss: np.ndarray | None = None
    mask_line: np.ndarray | None = None


def run_pygrappa(
    masked_kspace: np.ndarray,
    calib: np.ndarray,
    *,
    kernel_size: tuple[int, int],
    lamda: float,
) -> np.ndarray:
    """Run GRAPPA on [coil, H, W] arrays."""
    if pygrappa_grappa is None:
        raise ImportError(
            "pygrappa is required for multicoil GRAPPA preprocessing. "
            "Install the project dependencies before training."
        )
    masked_kspace = check_complex_coil_kspace(masked_kspace, name="masked_kspace")
    calib = check_complex_coil_kspace(calib, name="calib")
    recon = pygrappa_grappa(
        np.moveaxis(masked_kspace, 0, -1),
        np.moveaxis(calib, 0, -1),
        kernel_size=tuple(int(v) for v in kernel_size),
        coil_axis=-1,
        lamda=float(lamda),
        silent=True,
    )
    return np.moveaxis(np.asarray(recon), -1, 0).astype(np.complex64)


def _scale_complex_pair(
    noisy: np.ndarray,
    clean: np.ndarray,
    zero_filled: np.ndarray,
    *,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scale_source = np.abs(clean)
    finite = scale_source[np.isfinite(scale_source)]
    scale = float(np.percentile(finite, float(percentile))) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.mean(np.abs(clean)))
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    return (
        (noisy / scale).astype(np.complex64),
        (clean / scale).astype(np.complex64),
        (zero_filled / scale).astype(np.complex64),
        scale,
    )


def preprocess_multicoil_slice(
    raw_kspace: np.ndarray,
    *,
    config: PreprocessConfig,
    volume_name: str,
    slice_idx: int,
    target_rss: np.ndarray | None = None,
) -> MulticoilPreprocessResult:
    """Preprocess one multicoil slice into SNRAware native channels.

    The gmap path is intentionally pure: the stored gmap is all ones and the
    trainable correction adapter learns the effective gmap/scale correction.
    No sampling-based gmap estimation branch is implemented in this project.
    """
    raw_kspace = check_complex_coil_kspace(raw_kspace, name="raw_kspace")
    crop_size = (int(config.crop_size[0]), int(config.crop_size[1]))
    calib_center_fraction = float(config.calib_center_fraction)
    sample_seed = (
        seed_from_name(f"{volume_name}:{int(slice_idx)}:{int(config.sample_seed)}")
        if config.deterministic_mask_from_name
        else int(config.sample_seed) + int(slice_idx)
    )

    kspace = to_uniform_kspace(raw_kspace, crop_size)
    cov, whitening, cov_meta = estimate_noise_covariance(
        kspace,
        corner_fraction=float(config.cov_corner_fraction),
        shrinkage=float(config.cov_shrinkage),
        condition_max=float(config.cov_condition_max),
        eig_floor=float(config.eig_floor),
    )
    whitened = apply_coil_matrix(kspace, whitening.T)
    scc = estimate_scc_matrix(whitened, ncc=int(config.ncc), center_fraction=calib_center_fraction)
    compressed = apply_coil_matrix(whitened, scc)

    clean_coils = ifft2c_np(compressed)
    csm = estimate_csm(compressed, center_fraction=calib_center_fraction)
    clean_complex = sense_combine(clean_coils, csm)

    mask_line = build_1d_mask(
        compressed.shape[-1],
        acc_factor=int(config.acc_factor),
        center_fraction=float(config.center_fraction),
        seed=sample_seed,
        sampling_pattern=str(config.sampling_pattern),
    )
    masked = compressed.copy()
    masked[:, :, ~mask_line] = 0.0
    zero_filled_complex = sense_combine(ifft2c_np(masked), csm)

    calib = extract_center_calib(compressed, calib_center_fraction)
    grappa_kspace = run_pygrappa(
        masked,
        calib,
        kernel_size=(int(config.grappa_kernel[0]), int(config.grappa_kernel[1])),
        lamda=float(config.grappa_lambda),
    )
    noisy_complex = sense_combine(ifft2c_np(grappa_kspace), csm)
    noisy_complex, clean_complex, zero_filled_complex, scale = _scale_complex_pair(
        noisy_complex,
        clean_complex,
        zero_filled_complex,
        percentile=float(config.scale_percentile),
    )

    gmap = np.full(crop_size, float(config.gmap_value), dtype=np.float32)
    target = None if target_rss is None else center_crop_or_pad_real(target_rss, crop_size) / scale

    arrays = [
        noisy_complex.real,
        noisy_complex.imag,
        clean_complex.real,
        clean_complex.imag,
        zero_filled_complex.real,
        zero_filled_complex.imag,
        gmap,
    ]
    if not all(np.isfinite(arr).all() for arr in arrays):
        raise ValueError(f"Non-finite preprocessing output for {volume_name} slice {slice_idx}")

    metadata = {
        "volume_name": str(volume_name),
        "slice_idx": int(slice_idx),
        "original_shape": tuple(int(v) for v in raw_kspace.shape),
        "crop_size": crop_size,
        "acc_factor": int(config.acc_factor),
        "center_fraction": float(config.center_fraction),
        "calib_center_fraction": float(config.calib_center_fraction),
        "sampling_pattern": str(config.sampling_pattern),
        "ncoils_in": int(raw_kspace.shape[0]),
        "ncoils_cc": int(compressed.shape[0]),
        "scale": float(scale),
        "scale_percentile": float(config.scale_percentile),
        "mask_num_samples": int(mask_line.sum()),
        "mask_width": int(mask_line.size),
        "acs_lines": round(mask_line.size * float(config.calib_center_fraction)),
        "cov_trace": float(np.real(np.trace(cov)) / max(cov.shape[0], 1)),
        "gmap_mode": "ones_corrected",
        "gmap_value": float(config.gmap_value),
        **cov_meta,
    }
    return MulticoilPreprocessResult(
        noisy_complex=noisy_complex,
        clean_complex=clean_complex,
        gmap=gmap,
        metadata=metadata,
        zero_filled_complex=zero_filled_complex,
        target_rss=None if target is None else target.astype(np.float32),
        mask_line=mask_line.astype(bool),
    )
