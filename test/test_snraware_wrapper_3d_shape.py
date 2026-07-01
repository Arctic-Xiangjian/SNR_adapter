from pathlib import Path

import torch
import torch.nn as nn

from snraware.projects.mri.multicoil.config import CorrectionConfig, PatchShape3D
from snraware.projects.mri.multicoil.snraware_wrapper import (
    SNRAwareMulticoilWrapper,
    load_base_model_config,
)


class DummyBase3D(nn.Module):
    def forward(self, x):
        return x[:, 0:2] * 2.0


def test_wrapper_accepts_5d_2d_patch_by_default():
    patch = PatchShape3D(depth=1, height=64, width=64)
    wrapper = SNRAwareMulticoilWrapper(
        DummyBase3D(),
        CorrectionConfig(hidden_chans=4),
        patch_shape=patch,
    )
    x = torch.randn(1, 3, 1, 64, 64)
    x[:, 2] = 1.0

    y = wrapper(x)
    assert tuple(y.shape) == (1, 2, 1, 64, 64)
    assert torch.allclose(y, x[:, 0:2] * 2.0, atol=1.0e-6)

    try:
        wrapper(torch.randn(1, 3, 64, 64))
    except ValueError as exc:
        assert "[B, 3, D, H, W]" in str(exc)
    else:
        raise AssertionError("wrapper accepted a non-5D input")


def test_wrapper_accepts_explicit_3d_patch():
    patch = PatchShape3D(depth=16, height=64, width=64)
    wrapper = SNRAwareMulticoilWrapper(
        DummyBase3D(),
        CorrectionConfig(hidden_chans=4),
        patch_shape=patch,
    )
    x = torch.randn(1, 3, 16, 64, 64)
    x[:, 2] = 1.0

    y = wrapper(x)
    assert tuple(y.shape) == (1, 2, 16, 64, 64)
    assert torch.allclose(y, x[:, 0:2] * 2.0, atol=1.0e-6)


def test_base_config_uses_snraware_hwd_cutout(tmp_path: Path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text("dataset:\n  cutout_shape: [1, 1, 1]\n", encoding="utf-8")

    config = load_base_model_config(config_path, PatchShape3D(depth=1, height=64, width=64))
    assert list(config.dataset.cutout_shape) == [64, 64, 1]


def test_base_config_uses_explicit_3d_snraware_hwd_cutout(tmp_path: Path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text("dataset:\n  cutout_shape: [1, 1, 1]\n", encoding="utf-8")

    config = load_base_model_config(config_path, PatchShape3D(depth=16, height=64, width=64))
    assert list(config.dataset.cutout_shape) == [64, 64, 16]
