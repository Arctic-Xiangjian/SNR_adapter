import numpy as np

from snraware.projects.mri.multicoil import physics
from snraware.projects.mri.multicoil.config import PreprocessConfig


def test_preprocess_slice_contract_keeps_ones_gmap(monkeypatch):
    rng = np.random.default_rng(1234)
    raw = (
        rng.standard_normal((4, 40, 36), dtype=np.float32)
        + 1j * rng.standard_normal((4, 40, 36), dtype=np.float32)
    ).astype(np.complex64)

    def fake_grappa(masked_kspace, calib, *, kernel_size, lamda):
        return masked_kspace

    monkeypatch.setattr(physics, "run_pygrappa", fake_grappa)
    config = PreprocessConfig(
        crop_size=[32, 32],
        acc_factor=4,
        center_fraction=0.25,
        calib_center_fraction=0.25,
        ncc=4,
        cov_corner_fraction=0.25,
        gmap_value=1.0,
        cache_dir=None,
    )
    result = physics.preprocess_multicoil_slice(
        raw,
        config=config,
        volume_name="synthetic",
        slice_idx=3,
    )

    assert result.noisy_complex.shape == (32, 32)
    assert result.clean_complex.shape == (32, 32)
    assert result.gmap.shape == (32, 32)
    assert np.isfinite(result.noisy_complex.real).all()
    assert np.isfinite(result.clean_complex.imag).all()
    assert np.allclose(result.gmap, 1.0)
    assert np.isfinite(result.metadata["scale"])
    assert "whiten_mode" in result.metadata
    assert "cov_condition" in result.metadata
