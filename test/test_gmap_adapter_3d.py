import torch

from snraware.projects.mri.multicoil.config import CorrectionConfig
from snraware.projects.mri.multicoil.gmap_adapter import GFactorCorrectionAdapter3D


def test_gmap_adapter_zero_init_identity_and_gradients():
    torch.manual_seed(5)
    adapter = GFactorCorrectionAdapter3D(CorrectionConfig(hidden_chans=4))
    x = torch.randn(2, 3, 16, 64, 64)
    x[:, 2] = 1.0

    out = adapter(x)
    assert tuple(out.shape) == tuple(x.shape)
    assert torch.allclose(out[:, 0:2], x[:, 0:2], atol=1.0e-6)
    assert torch.allclose(out[:, 2:3], x[:, 2:3], atol=1.0e-6)
    assert adapter.last_stats is not None
    assert adapter.last_stats["complex_scale"] == 1.0
    assert adapter.last_stats["gmap_mean"] == 1.0

    loss = out[:, 2:3].sum()
    loss.backward()
    gmap_grads = [p.grad for p in adapter.gmap_unet.parameters() if p.grad is not None]
    assert any(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in gmap_grads)


def test_gmap_adapter_rejects_non_5d_input():
    adapter = GFactorCorrectionAdapter3D(CorrectionConfig(hidden_chans=4))
    x = torch.randn(2, 3, 64, 64)
    try:
        adapter(x)
    except ValueError as exc:
        assert "[B, 3, D, H, W]" in str(exc)
    else:
        raise AssertionError("adapter accepted a non-5D input")
