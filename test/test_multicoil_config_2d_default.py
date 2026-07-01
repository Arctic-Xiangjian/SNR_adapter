import pytest

from snraware.projects.mri.multicoil.config import (
    PatchShape3D,
    load_project_config,
)
from snraware.projects.mri.multicoil.trainer import _validate_checkpoint_patch

CONFIG_PATH = "configs/multicoil/fastmri_x8_cf004_partial05_gmap_ones.yaml"


def test_default_multicoil_config_is_2d_first():
    config = load_project_config(CONFIG_PATH)

    assert config.train.patch.depth == 1
    assert config.train.inference_overlap.depth == 0
    assert config.runtime.run_name == "fastmri_x8_cf004_partial05_2d_d1_gmap_ones"


def test_explicit_3d_depth_16_config_is_supported():
    config = load_project_config(
        CONFIG_PATH,
        overrides=["train.patch.depth=16", "train.inference_overlap.depth=8"],
    )

    assert config.train.patch.depth == 16
    assert config.train.inference_overlap.depth == 8


@pytest.mark.parametrize("depth", [2, 8, 15, 17])
def test_multicoil_config_rejects_unsupported_patch_depth(depth: int):
    with pytest.raises(ValueError, match=r"train\.patch\.depth supports only 1"):
        load_project_config(
            CONFIG_PATH,
            overrides=[f"train.patch.depth={depth}", "train.inference_overlap.depth=0"],
        )


def test_2d_multicoil_config_rejects_depth_overlap():
    with pytest.raises(ValueError, match=r"inference_overlap\.depth must be 0"):
        load_project_config(CONFIG_PATH, overrides=["train.inference_overlap.depth=1"])


def test_checkpoint_patch_validation_accepts_matching_2d_patch():
    payload = {"patch": {"depth": 1, "height": 64, "width": 64}}

    _validate_checkpoint_patch(payload, PatchShape3D(depth=1, height=64, width=64), "checkpoint.pth")


def test_checkpoint_patch_validation_rejects_3d_to_2d_resume():
    payload = {"patch": {"depth": 16, "height": 64, "width": 64}}

    with pytest.raises(RuntimeError, match="Checkpoint patch shape does not match"):
        _validate_checkpoint_patch(payload, PatchShape3D(depth=1, height=64, width=64), "checkpoint.pth")
