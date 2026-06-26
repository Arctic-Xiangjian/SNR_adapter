"""H5 dataset bridge for pure multicoil SNRAware training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import cache_path_for_slice, load_preprocess_cache, save_preprocess_cache
from .config import H5DataConfig, PreprocessConfig, SubsetConfig
from .physics import MulticoilPreprocessResult, preprocess_multicoil_slice


@dataclass(frozen=True)
class SliceRef:
    """Reference to one H5 volume slice."""

    path: Path
    volume_name: str
    slice_idx: int
    source_fingerprint: str


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime_ns)}"


def _expand_h5_roots(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root_text in roots:
        root = Path(root_text)
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.glob("*.h5")))
        else:
            raise FileNotFoundError(f"H5 root does not exist: {root}")
    return sorted(dict.fromkeys(files))


def _normalize_complex(arr: np.ndarray, *, complex_format: str) -> np.ndarray:
    if complex_format == "native":
        if not np.iscomplexobj(arr):
            raise ValueError("complex_format=native requires a complex-valued H5 dataset")
        return arr.astype(np.complex64, copy=False)
    if arr.shape[-1] != 2:
        raise ValueError("complex_format=real_imag_last requires last dimension of size 2")
    return (arr[..., 0].astype(np.float32) + 1j * arr[..., 1].astype(np.float32)).astype(np.complex64)


def _read_kspace_slice(handle: h5py.File, data_cfg: H5DataConfig, slice_idx: int) -> np.ndarray:
    data = handle[data_cfg.kspace_key]
    raw = np.take(data, int(slice_idx), axis=int(data_cfg.slice_axis))
    coil_axis = int(data_cfg.coil_axis)
    if coil_axis > int(data_cfg.slice_axis):
        coil_axis -= 1
    raw = np.moveaxis(np.asarray(raw), coil_axis, 0)
    return _normalize_complex(raw, complex_format=data_cfg.complex_format)


def _read_target_slice(handle: h5py.File, data_cfg: H5DataConfig, slice_idx: int) -> np.ndarray | None:
    if data_cfg.target_key in (None, "", "null"):
        return None
    if str(data_cfg.target_key) not in handle:
        return None
    target = np.take(handle[str(data_cfg.target_key)], int(slice_idx), axis=int(data_cfg.slice_axis))
    target = np.asarray(target)
    if np.iscomplexobj(target):
        target = np.abs(target)
    elif target.ndim >= 3 and target.shape[-1] == 2 and data_cfg.complex_format == "real_imag_last":
        target = np.abs(target[..., 0] + 1j * target[..., 1])
    return target.astype(np.float32, copy=False)


def _tensorize_result(result: MulticoilPreprocessResult) -> dict[str, Any]:
    noisy = np.stack(
        [result.noisy_complex.real, result.noisy_complex.imag, result.gmap],
        axis=0,
    ).astype(np.float32)
    clean = np.stack(
        [result.clean_complex.real, result.clean_complex.imag],
        axis=0,
    ).astype(np.float32)
    metadata = dict(result.metadata)
    volume_name = str(metadata.get("volume_name", "unknown_volume"))
    slice_idx = int(metadata.get("slice_idx", -1))
    metadata.setdefault("name", f"{volume_name}_slice_{slice_idx}")
    metadata.setdefault("volume_name", volume_name)
    metadata.setdefault("slice_idx", slice_idx)
    metadata.setdefault("mean", 0.0)
    metadata.setdefault("std", float(metadata.get("scale", 1.0)))
    if result.target_rss is not None:
        metadata["target_rss"] = torch.from_numpy(np.asarray(result.target_rss, dtype=np.float32)).contiguous()
    return {
        "noisy": torch.from_numpy(noisy).contiguous(),
        "clean": torch.from_numpy(clean).contiguous(),
        "metadata": metadata,
    }


class MulticoilH5Dataset(Dataset):
    """Dataset emitting [real, imag, ones-gmap] inputs and complex targets."""

    def __init__(
        self,
        data_config: H5DataConfig,
        preprocess_config: PreprocessConfig,
        *,
        split: str,
        subset: SubsetConfig | None = None,
        train_patch_size: list[int] | tuple[int, int] | None = None,
    ):
        super().__init__()
        self.data_config = data_config
        self.preprocess_config = preprocess_config
        self.split = split
        self.train_patch_size = None if train_patch_size is None else tuple(int(v) for v in train_patch_size)

        files = _expand_h5_roots(data_config.roots)
        if not files:
            raise ValueError(f"No H5 files found for split={split}")
        refs: list[SliceRef] = []
        for path in files:
            with h5py.File(path, "r") as handle:
                if data_config.kspace_key not in handle:
                    raise KeyError(f"{path} is missing kspace_key={data_config.kspace_key!r}")
                num_slices = int(handle[data_config.kspace_key].shape[int(data_config.slice_axis)])
            if data_config.max_slices is not None:
                num_slices = min(num_slices, int(data_config.max_slices))
            source_fingerprint = _file_fingerprint(path)
            refs.extend(
                SliceRef(
                    path=path,
                    volume_name=path.stem,
                    slice_idx=slice_idx,
                    source_fingerprint=source_fingerprint,
                )
                for slice_idx in range(num_slices)
            )

        refs = self._apply_volume_sample_fraction(refs, data_config)
        if subset is not None and split == "train":
            refs = self._apply_training_subset(refs, subset)
        self.refs = refs
        self.dataset_info = {
            "split": split,
            "num_slices": len(self.refs),
            "num_volumes": len({ref.volume_name for ref in self.refs}),
            "roots": list(data_config.roots),
            "format": data_config.format,
        }

    @staticmethod
    def _apply_volume_sample_fraction(refs: list[SliceRef], data_config: H5DataConfig) -> list[SliceRef]:
        fraction = data_config.volume_sample_fraction
        if fraction is None:
            return refs
        if not (0.0 < float(fraction) <= 1.0):
            raise ValueError(f"volume_sample_fraction must be in (0, 1], got {fraction}")
        volumes = sorted({ref.volume_name for ref in refs})
        rng = random.Random(int(data_config.volume_sample_seed))
        rng.shuffle(volumes)
        selected = set(volumes[: max(1, round(len(volumes) * float(fraction)))])
        return [ref for ref in refs if ref.volume_name in selected]

    @staticmethod
    def _apply_training_subset(refs: list[SliceRef], subset: SubsetConfig) -> list[SliceRef]:
        if subset.mode == "none":
            return refs
        rng = random.Random(int(subset.seed))
        if subset.mode == "random_slice":
            indexed = list(enumerate(refs))
            rng.shuffle(indexed)
            selected_indices = sorted(index for index, _ in indexed[: max(1, round(len(refs) * float(subset.fraction)))])
            return [refs[index] for index in selected_indices]
        volumes = sorted({ref.volume_name for ref in refs})
        rng.shuffle(volumes)
        selected = set(volumes[: max(1, round(len(volumes) * float(subset.fraction)))])
        return [ref for ref in refs if ref.volume_name in selected]

    def __len__(self) -> int:
        return len(self.refs)

    def _load_or_preprocess(self, ref: SliceRef) -> MulticoilPreprocessResult:
        cache_path = cache_path_for_slice(
            cache_dir=self.preprocess_config.cache_dir,
            config=self.preprocess_config,
            volume_name=ref.volume_name,
            slice_idx=ref.slice_idx,
            source_fingerprint=ref.source_fingerprint,
        )
        if cache_path is not None:
            cached = load_preprocess_cache(cache_path)
            if cached is not None:
                return cached

        with h5py.File(ref.path, "r") as handle:
            raw_kspace = _read_kspace_slice(handle, self.data_config, ref.slice_idx)
            target = _read_target_slice(handle, self.data_config, ref.slice_idx)
        result = preprocess_multicoil_slice(
            raw_kspace,
            config=self.preprocess_config,
            volume_name=ref.volume_name,
            slice_idx=ref.slice_idx,
            target_rss=target,
        )
        result.metadata["source_path"] = str(ref.path)
        result.metadata["source_fingerprint"] = ref.source_fingerprint
        if cache_path is not None:
            save_preprocess_cache(cache_path, result)
        return result

    def _maybe_crop_patch(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.split != "train" or self.train_patch_size is None:
            return sample
        patch_h, patch_w = self.train_patch_size
        noisy = sample["noisy"]
        clean = sample["clean"]
        full_h, full_w = clean.shape[-2:]
        if patch_h > full_h or patch_w > full_w:
            raise ValueError(f"train_patch_size {self.train_patch_size} must fit inside {(full_h, full_w)}")
        top = int(torch.randint(0, full_h - patch_h + 1, ()).item())
        left = int(torch.randint(0, full_w - patch_w + 1, ()).item())
        out = dict(sample)
        out["noisy"] = noisy[..., top : top + patch_h, left : left + patch_w]
        out["clean"] = clean[..., top : top + patch_h, left : left + patch_w]
        out["metadata"] = dict(sample["metadata"])
        out["metadata"]["patch_top"] = top
        out["metadata"]["patch_left"] = left
        out["metadata"]["patch_size"] = [patch_h, patch_w]
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[int(index)]
        sample = _tensorize_result(self._load_or_preprocess(ref))
        return self._maybe_crop_patch(sample)


def collate_multicoil_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors while keeping metadata as a list of plain dicts."""
    return {
        "noisy": torch.stack([sample["noisy"] for sample in samples], dim=0),
        "clean": torch.stack([sample["clean"] for sample in samples], dim=0),
        "metadata": [sample["metadata"] for sample in samples],
    }
