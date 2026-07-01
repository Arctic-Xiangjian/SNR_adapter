import torch

from snraware.projects.mri.multicoil.config import CorrectionConfig
from snraware.projects.mri.multicoil.gmap_adapter import GFactorCorrectionAdapter3D, GFactorUNet3D


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


def test_gmap_unet_downsampling_is_spatial_only():
    unet = GFactorUNet3D(in_chans=3, hidden_chans=4)
    assert unet.down1.stride == (1, 2, 2)
    assert unet.down2.stride == (1, 2, 2)

    x = torch.randn(1, 3, 5, 32, 32)
    with torch.no_grad():
        x1 = unet.enc1(x)
        x2_down = unet.down1(x1)
        x2 = unet.enc2(x2_down)
        x3_down = unet.down2(x2)
        out = unet(x)

    assert tuple(x2_down.shape[-3:]) == (5, 16, 16)
    assert tuple(x3_down.shape[-3:]) == (5, 8, 8)
    assert tuple(out.shape) == (1, 1, 5, 32, 32)
