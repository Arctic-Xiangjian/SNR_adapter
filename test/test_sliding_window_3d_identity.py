import torch
import torch.nn as nn

from snraware.projects.mri.multicoil.config import OverlapShape3D, PatchShape3D
from snraware.projects.mri.multicoil.sliding_window import (
    patch_positions,
    predict_sliding_window_3d,
)


class DummyIdentity3D(nn.Module):
    def forward(self, x):
        return x[:, 0:2]


def test_patch_positions_cover_axis():
    assert patch_positions(16, 16, 8) == [0]
    assert patch_positions(20, 16, 8) == [0, 4]
    assert patch_positions(384, 64, 16)[0] == 0
    assert patch_positions(384, 64, 16)[-1] == 320


def test_sliding_window_identity_full_384_volume():
    torch.manual_seed(7)
    noisy = torch.randn(1, 3, 16, 384, 384)
    pred = predict_sliding_window_3d(
        DummyIdentity3D(),
        noisy,
        patch=PatchShape3D(depth=16, height=64, width=64),
        overlap=OverlapShape3D(depth=8, height=16, width=16),
        patch_batch_size=16,
    )

    assert tuple(pred.shape) == (1, 2, 16, 384, 384)
    assert torch.allclose(pred, noisy[:, 0:2], atol=1.0e-5)


def test_sliding_window_identity_full_volume_2d_depth_one_patches():
    torch.manual_seed(11)
    noisy = torch.randn(1, 3, 5, 384, 384)
    pred = predict_sliding_window_3d(
        DummyIdentity3D(),
        noisy,
        patch=PatchShape3D(depth=1, height=64, width=64),
        overlap=OverlapShape3D(depth=0, height=16, width=16),
        patch_batch_size=16,
    )

    assert tuple(pred.shape) == (1, 2, 5, 384, 384)
    assert torch.allclose(pred, noisy[:, 0:2], atol=1.0e-5)
