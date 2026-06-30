"""Transparent preprocessing cache for multicoil slices."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PreprocessConfig
from .physics import MulticoilPreprocessResult


def preprocess_fingerprint(config: PreprocessConfig) -> str:
    """Stable hash for cache-relevant preprocessing config."""
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cache_path_for_slice(
    *,
    cache_dir: str | Path | None,
    config: PreprocessConfig,
    volume_name: str,
    slice_idx: int,
    source_fingerprint: str,
) -> Path | None:
    """Return cache path for a slice, or None when caching is disabled."""
    if cache_dir in (None, "", "null"):
        return None
    key = "|".join(
        [
            str(config.cache_version),
            preprocess_fingerprint(config),
            str(volume_name),
            str(int(slice_idx)),
            str(source_fingerprint),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{Path(volume_name).stem}_slice{int(slice_idx):04d}_{digest}.npz"


def save_preprocess_cache(path: Path, result: MulticoilPreprocessResult) -> None:
    """Write one preprocessed slice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "noisy_real": result.noisy_complex.real.astype(np.float32),
        "noisy_imag": result.noisy_complex.imag.astype(np.float32),
        "clean_real": result.clean_complex.real.astype(np.float32),
        "clean_imag": result.clean_complex.imag.astype(np.float32),
        "gmap": result.gmap.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(result.metadata, sort_keys=True, default=str)),
    }
    if result.zero_filled_complex is not None:
        payload["zero_filled_real"] = result.zero_filled_complex.real.astype(np.float32)
        payload["zero_filled_imag"] = result.zero_filled_complex.imag.astype(np.float32)
    if result.target_rss is not None:
        payload["target_rss"] = result.target_rss.astype(np.float32)
    if result.mask_line is not None:
        payload["mask_line"] = result.mask_line.astype(bool)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_preprocess_cache(path: Path) -> MulticoilPreprocessResult | None:
    """Load one preprocessed slice if present."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata_json"].item()))
            zero_filled = None
            if "zero_filled_real" in cached and "zero_filled_imag" in cached:
                zero_filled = cached["zero_filled_real"].astype(np.float32) + 1j * cached[
                    "zero_filled_imag"
                ].astype(np.float32)
            return MulticoilPreprocessResult(
                noisy_complex=cached["noisy_real"].astype(np.float32)
                + 1j * cached["noisy_imag"].astype(np.float32),
                clean_complex=cached["clean_real"].astype(np.float32)
                + 1j * cached["clean_imag"].astype(np.float32),
                gmap=cached["gmap"].astype(np.float32),
                metadata=metadata,
                zero_filled_complex=zero_filled,
                target_rss=cached["target_rss"].astype(np.float32) if "target_rss" in cached else None,
                mask_line=cached["mask_line"].astype(bool) if "mask_line" in cached else None,
            )
    except (OSError, ValueError, zipfile.BadZipFile):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
