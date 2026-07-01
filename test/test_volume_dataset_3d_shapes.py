import h5py
import numpy as np

from snraware.projects.mri.multicoil.config import H5DataConfig, PatchShape3D, PreprocessConfig
from snraware.projects.mri.multicoil.physics import MulticoilPreprocessResult
from snraware.projects.mri.multicoil.volume_dataset import (
    MulticoilVolumeDataset,
    collate_multicoil_batch,
)


def _make_h5(path):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("kspace", shape=(20, 2, 4, 4), dtype=np.complex64)


def _fake_result(slice_idx: int) -> MulticoilPreprocessResult:
    value = np.float32(slice_idx + 1)
    noisy = np.full((384, 384), value + 1j * value, dtype=np.complex64)
    clean = np.full((384, 384), value * 2 + 1j * value * 0.5, dtype=np.complex64)
    gmap = np.ones((384, 384), dtype=np.float32)
    return MulticoilPreprocessResult(
        noisy_complex=noisy,
        clean_complex=clean,
        gmap=gmap,
        metadata={"scale": float(value), "slice_idx": int(slice_idx)},
    )


def test_train_dataset_returns_contiguous_3d_window(monkeypatch, tmp_path):
    h5_path = tmp_path / "volume_a.h5"
    _make_h5(h5_path)

    def fake_load(self, ref):
        return _fake_result(ref.slice_idx)

    monkeypatch.setattr(MulticoilVolumeDataset, "_load_or_preprocess", fake_load)
    dataset = MulticoilVolumeDataset(
        H5DataConfig(roots=[str(h5_path)]),
        PreprocessConfig(cache_dir=None),
        split="train",
        patch_shape=PatchShape3D(depth=16, height=64, width=64),
    )
    sample = dataset[0]
    batch = collate_multicoil_batch([sample, sample])

    assert tuple(sample["noisy"].shape) == (3, 16, 64, 64)
    assert tuple(sample["clean"].shape) == (2, 16, 64, 64)
    assert tuple(batch["noisy"].shape) == (2, 3, 16, 64, 64)
    assert tuple(batch["clean"].shape) == (2, 2, 16, 64, 64)
    z_indices = sample["metadata"]["z_indices"]
    assert z_indices == list(range(z_indices[0], z_indices[0] + 16))
    assert len(sample["metadata"]["slice_scales"]) == 16
    assert sample["metadata"]["volume_name"] == "volume_a"
    assert 0 <= sample["metadata"]["patch_top"] <= 320
    assert 0 <= sample["metadata"]["patch_left"] <= 320


def test_train_dataset_returns_single_slice_2d_window(monkeypatch, tmp_path):
    h5_path = tmp_path / "volume_2d.h5"
    _make_h5(h5_path)

    def fake_load(self, ref):
        return _fake_result(ref.slice_idx)

    monkeypatch.setattr(MulticoilVolumeDataset, "_load_or_preprocess", fake_load)
    dataset = MulticoilVolumeDataset(
        H5DataConfig(roots=[str(h5_path)]),
        PreprocessConfig(cache_dir=None),
        split="train",
        patch_shape=PatchShape3D(depth=1, height=64, width=64),
    )
    sample = dataset[0]
    batch = collate_multicoil_batch([sample, sample])

    assert tuple(sample["noisy"].shape) == (3, 1, 64, 64)
    assert tuple(sample["clean"].shape) == (2, 1, 64, 64)
    assert tuple(batch["noisy"].shape) == (2, 3, 1, 64, 64)
    assert tuple(batch["clean"].shape) == (2, 2, 1, 64, 64)
    assert len(sample["metadata"]["z_indices"]) == 1
    assert len(sample["metadata"]["slice_scales"]) == 1
    assert sample["metadata"]["volume_name"] == "volume_2d"
    assert 0 <= sample["metadata"]["patch_top"] <= 320
    assert 0 <= sample["metadata"]["patch_left"] <= 320


def test_val_dataset_returns_full_volume(monkeypatch, tmp_path):
    h5_path = tmp_path / "volume_b.h5"
    _make_h5(h5_path)

    def fake_load(self, ref):
        return _fake_result(ref.slice_idx)

    monkeypatch.setattr(MulticoilVolumeDataset, "_load_or_preprocess", fake_load)
    dataset = MulticoilVolumeDataset(
        H5DataConfig(roots=[str(h5_path)]),
        PreprocessConfig(cache_dir=None),
        split="val",
        patch_shape=PatchShape3D(depth=16, height=64, width=64),
    )
    sample = dataset[0]
    batch = collate_multicoil_batch([sample])

    assert tuple(sample["noisy"].shape) == (3, 20, 384, 384)
    assert tuple(sample["clean"].shape) == (2, 20, 384, 384)
    assert tuple(batch["noisy"].shape) == (1, 3, 20, 384, 384)
    assert tuple(batch["clean"].shape) == (1, 2, 20, 384, 384)
    assert sample["metadata"]["z_indices"] == list(range(20))
    assert len(sample["metadata"]["slice_scales"]) == 20
