"""Low-level multicoil preprocessing primitives."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

EPS = 1.0e-8


def seed_from_name(name: str) -> int:
    """Stable integer seed from volume/slice identity."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def check_complex_coil_kspace(x: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and cast a [coil, height, width] complex array."""
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [coil, H, W], got {arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)
    return arr.astype(np.complex64, copy=False)


def fft2c_np(image: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2D FFT over the last two axes."""
    image = np.asarray(image)
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)


def ifft2c_np(kspace: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2D IFFT over the last two axes."""
    kspace = np.asarray(kspace)
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)


def _center_slices(current: int, target: int) -> tuple[slice, slice]:
    if current >= target:
        start = (current - target) // 2
        return slice(start, start + target), slice(0, target)
    start = (target - current) // 2
    return slice(0, current), slice(start, start + current)


def center_crop_or_pad_image(image: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Center crop or zero-pad complex coil image data to crop_size."""
    image = check_complex_coil_kspace(image, name="image")
    height, width = int(crop_size[0]), int(crop_size[1])
    out = np.zeros((image.shape[0], height, width), dtype=np.complex64)
    src_h, dst_h = _center_slices(image.shape[-2], height)
    src_w, dst_w = _center_slices(image.shape[-1], width)
    out[:, dst_h, dst_w] = image[:, src_h, src_w]
    return out


def center_crop_or_pad_real(image: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Center crop or zero-pad real image data to crop_size."""
    arr = np.asarray(image, dtype=np.float32)
    height, width = int(crop_size[0]), int(crop_size[1])
    out = np.zeros((height, width), dtype=np.float32)
    src_h, dst_h = _center_slices(arr.shape[-2], height)
    src_w, dst_w = _center_slices(arr.shape[-1], width)
    out[dst_h, dst_w] = arr[src_h, src_w]
    return out


def to_uniform_kspace(kspace: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Crop/pad in image domain first, then return k-space on the target grid."""
    kspace = check_complex_coil_kspace(kspace, name="kspace")
    image = ifft2c_np(kspace)
    cropped = center_crop_or_pad_image(image, crop_size)
    return fft2c_np(cropped)


def build_random_1d_mask(
    width: int,
    *,
    acc_factor: int,
    center_fraction: float,
    seed: int,
) -> np.ndarray:
    """Build deterministic random Cartesian mask with an ACS center."""
    width = int(width)
    nsamp_target = max(1, round(width / int(acc_factor)))
    nsamp_center = min(max(1, round(width * float(center_fraction))), width)
    nsamp_target = max(nsamp_target, nsamp_center)

    mask = np.zeros(width, dtype=bool)
    center_from = width // 2 - nsamp_center // 2
    center_to = center_from + nsamp_center
    mask[center_from:center_to] = True

    remaining = nsamp_target - int(mask.sum())
    if remaining > 0:
        center_indices = np.arange(center_from, center_to)
        candidates = np.setdiff1d(np.arange(width), center_indices, assume_unique=True)
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)
        mask[selected] = True
    return mask


def build_uniform_1d_mask(
    width: int,
    *,
    acc_factor: int,
    center_fraction: float,
) -> np.ndarray:
    """Build deterministic equispaced Cartesian mask with an ACS center."""
    width = int(width)
    nsamp_target = max(1, round(width / int(acc_factor)))
    nsamp_center = min(max(1, round(width * float(center_fraction))), width)
    nsamp_target = max(nsamp_target, nsamp_center)

    mask = np.zeros(width, dtype=bool)
    center_from = width // 2 - nsamp_center // 2
    center_to = center_from + nsamp_center
    mask[center_from:center_to] = True

    remaining = nsamp_target - int(mask.sum())
    if remaining > 0:
        center_indices = np.arange(center_from, center_to)
        candidates = np.setdiff1d(np.arange(width), center_indices, assume_unique=True)
        positions = np.linspace(0, len(candidates) - 1, num=min(remaining, len(candidates)))
        mask[candidates[np.rint(positions).astype(np.int64)]] = True
    return mask


def build_1d_mask(
    width: int,
    *,
    acc_factor: int,
    center_fraction: float,
    seed: int,
    sampling_pattern: str,
) -> np.ndarray:
    """Build the requested 1D Cartesian undersampling mask."""
    pattern = sampling_pattern.lower().replace("-", "_")
    if pattern in {"uniform", "equispaced", "regular"}:
        return build_uniform_1d_mask(
            width,
            acc_factor=acc_factor,
            center_fraction=center_fraction,
        )
    if pattern in {"random", "random1d"}:
        return build_random_1d_mask(
            width,
            acc_factor=acc_factor,
            center_fraction=center_fraction,
            seed=seed,
        )
    raise ValueError(f"Unsupported sampling_pattern={sampling_pattern!r}")


def _corner_noise_samples(kspace: np.ndarray, corner_fraction: float) -> np.ndarray:
    coils, height, width = kspace.shape
    corner_h = min(max(4, round(height * float(corner_fraction))), max(1, height // 2))
    corner_w = min(max(4, round(width * float(corner_fraction))), max(1, width // 2))
    corners = [
        kspace[:, :corner_h, :corner_w],
        kspace[:, :corner_h, -corner_w:],
        kspace[:, -corner_h:, :corner_w],
        kspace[:, -corner_h:, -corner_w:],
    ]
    return np.concatenate([corner.reshape(coils, -1) for corner in corners], axis=1)


def estimate_noise_covariance(
    kspace: np.ndarray,
    *,
    corner_fraction: float,
    shrinkage: float,
    condition_max: float,
    eig_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate coil covariance from high-frequency corners and build whitening."""
    kspace = check_complex_coil_kspace(kspace, name="kspace")
    coils = int(kspace.shape[0])
    identity = np.eye(coils, dtype=np.complex64)
    metadata: dict[str, Any] = {
        "whiten_mode": "estimated_corner",
        "cov_fallback": False,
        "cov_condition": float("nan"),
    }
    try:
        samples = _corner_noise_samples(kspace, corner_fraction)
        samples = samples - samples.mean(axis=1, keepdims=True)
        if samples.shape[1] < coils:
            raise ValueError("not enough corner noise samples")
        cov = samples @ samples.conj().T / max(samples.shape[1] - 1, 1)
        cov = 0.5 * (cov + cov.conj().T)
        trace_scale = float(np.real(np.trace(cov)) / max(coils, 1))
        if not np.isfinite(trace_scale) or trace_scale <= 0:
            raise ValueError("invalid covariance trace")
        cov = (1.0 - shrinkage) * cov + shrinkage * trace_scale * identity
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(np.real(eigvals), max(eig_floor * trace_scale, eig_floor))
        condition = float(eigvals.max() / max(eigvals.min(), EPS))
        metadata["cov_condition"] = condition
        if not np.isfinite(condition) or condition > condition_max:
            raise ValueError(f"covariance condition {condition:.3g} exceeds limit")
        cov = (eigvecs * eigvals) @ eigvecs.conj().T
        chol = np.linalg.cholesky(cov)
        whitening = np.linalg.solve(chol, identity).astype(np.complex64)
        return cov.astype(np.complex64), whitening, metadata
    except Exception as exc:
        metadata.update(
            {
                "whiten_mode": "identity_fallback",
                "cov_fallback": True,
                "cov_fallback_reason": str(exc),
                "cov_condition": float("inf"),
            }
        )
        return identity, identity, metadata


def apply_coil_matrix(kspace: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a coil-space matrix to k-space shaped [coil, H, W]."""
    kspace = check_complex_coil_kspace(kspace, name="kspace")
    matrix = np.asarray(matrix, dtype=np.complex64)
    if matrix.shape[0] != kspace.shape[0]:
        raise ValueError(f"coil matrix shape {matrix.shape} is incompatible with kspace {kspace.shape}")
    return np.einsum("chw,cn->nhw", kspace, matrix, optimize=True).astype(np.complex64)


def estimate_scc_matrix(kspace: np.ndarray, *, ncc: int, center_fraction: float) -> np.ndarray:
    """Estimate slice-wise coil compression matrix from centered calibration samples."""
    kspace = check_complex_coil_kspace(kspace, name="kspace")
    coils, _height, width = kspace.shape
    ncc = min(max(1, int(ncc)), coils)
    acs_width = max(1, round(width * float(center_fraction)))
    start = width // 2 - acs_width // 2
    calib = kspace[:, :, start : start + acs_width]
    samples = np.moveaxis(calib, 0, -1).reshape(-1, coils)
    samples = samples[np.linalg.norm(samples, axis=1) > 0]
    if samples.shape[0] < ncc:
        return np.eye(coils, ncc, dtype=np.complex64)
    _u, _s, vh = np.linalg.svd(samples, full_matrices=False)
    return vh.conj().T[:, :ncc].astype(np.complex64)


def extract_center_calib(kspace: np.ndarray, center_fraction: float) -> np.ndarray:
    """Extract the centered ACS region for GRAPPA and CSM estimation."""
    kspace = check_complex_coil_kspace(kspace, name="kspace")
    width = int(kspace.shape[-1])
    acs_width = max(1, round(width * float(center_fraction)))
    start = width // 2 - acs_width // 2
    return kspace[:, :, start : start + acs_width].copy()


def hamming_windowed_calib(kspace: np.ndarray, center_fraction: float) -> np.ndarray:
    """Put a Hamming-windowed ACS block onto the full k-space grid."""
    kspace = check_complex_coil_kspace(kspace, name="kspace")
    _coils, _height, width = kspace.shape
    calib_width = max(1, round(width * float(center_fraction)))
    start = width // 2 - calib_width // 2
    calib = np.zeros_like(kspace, dtype=np.complex64)
    window = np.hamming(calib_width).astype(np.float32)
    calib[:, :, start : start + calib_width] = kspace[:, :, start : start + calib_width] * window.reshape(1, 1, -1)
    return calib


def estimate_csm(kspace: np.ndarray, *, center_fraction: float) -> np.ndarray:
    """Estimate single-map coil sensitivities using Hamming-windowed ACS data."""
    lowres = ifft2c_np(hamming_windowed_calib(kspace, center_fraction))
    rss = np.sqrt(np.sum(np.abs(lowres) ** 2, axis=0, keepdims=True))
    csm = lowres / np.maximum(rss, EPS)
    csm[:, rss.squeeze(0) <= EPS] = 0.0
    return csm.astype(np.complex64)


def sense_combine(coil_images: np.ndarray, csm: np.ndarray) -> np.ndarray:
    """SENSE combine coils using the same convention as the validated old run."""
    coil_images = check_complex_coil_kspace(coil_images, name="coil_images")
    csm = check_complex_coil_kspace(csm, name="csm")
    if coil_images.shape != csm.shape:
        raise ValueError(f"coil_images shape {coil_images.shape} must match csm shape {csm.shape}")
    denom = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0))
    numer = np.sum(coil_images * np.conj(csm), axis=0)
    out = np.zeros_like(numer, dtype=np.complex64)
    valid = denom > EPS
    out[valid] = numer[valid] / denom[valid]
    return out.astype(np.complex64)
