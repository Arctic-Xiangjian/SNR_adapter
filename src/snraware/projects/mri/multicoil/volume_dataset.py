"""Volume-stacking dataset for 3D-only multicoil SNRAware training."""

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
from .config import H5DataConfig, PatchShape3D, PreprocessConfig, SubsetConfig
from .physics import MulticoilPreprocessResult, preprocess_multicoil_slice


@dataclass(frozen=True)
class SliceRef:
    """Reference to one H5 volume slice."""

    path: Path
    volume_name: str
    slice_idx: int
    source_fingerprint: str


@dataclass(frozen=True)
class VolumeRef:
    """Reference to a full source volume."""

    volume_name: str
    slices: tuple[SliceRef, ...]


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime_ns)}"


def _expand_h5_roots(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root_text in roots:
        root = Path(root_text).expanduser()
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


def _tensorize_slice_result(result: MulticoilPreprocessResult) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    noisy = np.stack(
        [result.noisy_complex.real, result.noisy_complex.imag, result.gmap],
        axis=0,
    ).astype(np.float32)
    clean = np.stack(
        [result.clean_complex.real, result.clean_complex.imag],
        axis=0,
    ).astype(np.float32)
    metadata = dict(result.metadata)
    metadata.setdefault("scale", 1.0)
    return (
        torch.from_numpy(noisy).contiguous(),
        torch.from_numpy(clean).contiguous(),
        metadata,
    )


class MulticoilVolumeDataset(Dataset):
    """Emit 3D training windows or full validation/test volumes."""

    def __init__(
        self,
        data_config: H5DataConfig,
        preprocess_config: PreprocessConfig,
        *,
        split: str,
        patch_shape: PatchShape3D,
        subset: SubsetConfig | None = None,
    ):
        super().__init__()
        self.data_config = data_config
        self.preprocess_config = preprocess_config
        self.split = str(split)
        self.patch_shape = PatchShape3D.from_value(patch_shape)

        files = _expand_h5_roots(data_config.roots)
        if not files:
            raise ValueError(f"No H5 files found for split={split}")
        volumes: list[VolumeRef] = []
        for path in files:
            with h5py.File(path, "r") as handle:
                if data_config.kspace_key not in handle:
                    raise KeyError(f"{path} is missing kspace_key={data_config.kspace_key!r}")
                num_slices = int(handle[data_config.kspace_key].shape[int(data_config.slice_axis)])
            if data_config.max_slices is not None:
                num_slices = min(num_slices, int(data_config.max_slices))
            source_fingerprint = _file_fingerprint(path)
            slices = tuple(
                SliceRef(
                    path=path,
                    volume_name=path.stem,
                    slice_idx=slice_idx,
                    source_fingerprint=source_fingerprint,
                )
                for slice_idx in range(num_slices)
            )
            if len(slices) < self.patch_shape.depth:
                raise ValueError(
                    f"Volume {path} has D={len(slices)} but train.patch.depth={self.patch_shape.depth}"
                )
            volumes.append(VolumeRef(volume_name=path.stem, slices=slices))

        volumes = self._apply_volume_sample_fraction(volumes, data_config)
        if subset is not None and self.split == "train":
            volumes = self._apply_training_subset(volumes, subset)
        self.volumes = volumes
        self.dataset_info = {
            "split": self.split,
            "num_volumes": len(self.volumes),
            "num_slices": sum(len(volume.slices) for volume in self.volumes),
            "roots": list(data_config.roots),
            "format": data_config.format,
            "shape_contract": "[B,C,D,H,W]",
            "patch": {
                "depth": self.patch_shape.depth,
                "height": self.patch_shape.height,
                "width": self.patch_shape.width,
            },
        }

    @staticmethod
    def _apply_volume_sample_fraction(
        volumes: list[VolumeRef],
        data_config: H5DataConfig,
    ) -> list[VolumeRef]:
        fraction = data_config.volume_sample_fraction
        if fraction is None:
            return volumes
        if not (0.0 < float(fraction) <= 1.0):
            raise ValueError(f"volume_sample_fraction must be in (0, 1], got {fraction}")
        rng = random.Random(int(data_config.volume_sample_seed))
        shuffled = list(volumes)
        rng.shuffle(shuffled)
        selected_names = {
            volume.volume_name for volume in shuffled[: max(1, round(len(shuffled) * float(fraction)))]
        }
        return [volume for volume in volumes if volume.volume_name in selected_names]

    @staticmethod
    def _apply_training_subset(volumes: list[VolumeRef], subset: SubsetConfig) -> list[VolumeRef]:
        if subset.mode == "none":
            return volumes
        if subset.mode == "random_slice":
            raise ValueError("The 3D multicoil pipeline does not support random_slice subsets")
        rng = random.Random(int(subset.seed))
        shuffled = list(volumes)
        rng.shuffle(shuffled)
        selected_names = {
            volume.volume_name for volume in shuffled[: max(1, round(len(shuffled) * float(subset.fraction)))]
        }
        return [volume for volume in volumes if volume.volume_name in selected_names]

    def __len__(self) -> int:
        return len(self.volumes)

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

    def _stack_refs(self, refs: tuple[SliceRef, ...]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        noisy_slices: list[torch.Tensor] = []
        clean_slices: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        for ref in refs:
            noisy, clean, slice_metadata = _tensorize_slice_result(self._load_or_preprocess(ref))
            noisy_slices.append(noisy)
            clean_slices.append(clean)
            metadata.append(slice_metadata)
        noisy_volume = torch.stack(noisy_slices, dim=1)
        clean_volume = torch.stack(clean_slices, dim=1)
        return noisy_volume, clean_volume, metadata

    def _metadata_for_refs(
        self,
        volume: VolumeRef,
        refs: tuple[SliceRef, ...],
        slice_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "volume_name": volume.volume_name,
            "z_indices": [int(ref.slice_idx) for ref in refs],
            "slice_scales": [float(item.get("scale", item.get("std", 1.0))) for item in slice_metadata],
        }

    def _train_sample(self, volume: VolumeRef) -> dict[str, Any]:
        depth = self.patch_shape.depth
        max_start = len(volume.slices) - depth
        z_start = int(torch.randint(0, max_start + 1, ()).item())
        refs = volume.slices[z_start : z_start + depth]
        noisy, clean, slice_metadata = self._stack_refs(refs)
        full_h, full_w = clean.shape[-2:]
        patch_h, patch_w = self.patch_shape.height, self.patch_shape.width
        if patch_h > full_h or patch_w > full_w:
            raise ValueError(f"train.patch {(depth, patch_h, patch_w)} must fit inside {(depth, full_h, full_w)}")
        top = int(torch.randint(0, full_h - patch_h + 1, ()).item())
        left = int(torch.randint(0, full_w - patch_w + 1, ()).item())
        metadata = self._metadata_for_refs(volume, refs, slice_metadata)
        metadata.update(
            {
                "z_start": int(refs[0].slice_idx),
                "patch_top": top,
                "patch_left": left,
                "patch": {
                    "depth": depth,
                    "height": patch_h,
                    "width": patch_w,
                },
            }
        )
        return {
            "noisy": noisy[:, :, top : top + patch_h, left : left + patch_w],
            "clean": clean[:, :, top : top + patch_h, left : left + patch_w],
            "metadata": metadata,
        }

    def _volume_sample(self, volume: VolumeRef) -> dict[str, Any]:
        noisy, clean, slice_metadata = self._stack_refs(volume.slices)
        metadata = self._metadata_for_refs(volume, volume.slices, slice_metadata)
        return {"noisy": noisy, "clean": clean, "metadata": metadata}

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume = self.volumes[int(index)]
        if self.split == "train":
            return self._train_sample(volume)
        return self._volume_sample(volume)


def collate_multicoil_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors while keeping metadata as a list of plain dicts."""
    return {
        "noisy": torch.stack([sample["noisy"] for sample in samples], dim=0),
        "clean": torch.stack([sample["clean"] for sample in samples], dim=0),
        "metadata": [sample["metadata"] for sample in samples],
    }
