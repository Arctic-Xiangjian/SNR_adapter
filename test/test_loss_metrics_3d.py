import numpy as np
import torch
import torch.nn as nn

from snraware.projects.mri.multicoil.metrics import (
    complex_magnitude,
    compute_volume_metrics,
    current_magnitude_max,
    restore_magnitude_volumes,
)
from snraware.projects.mri.multicoil.trainer import MulticoilFineTuneTrainer


def test_5d_magnitude_and_current_max_scale_shapes():
    pred = torch.zeros(2, 2, 16, 8, 8)
    pred[:, 0] = 3.0
    pred[:, 1] = 4.0
    noisy = torch.zeros(2, 3, 16, 8, 8)
    noisy[:, 0] = 0.6
    noisy[:, 1] = 0.8
    noisy[:, 0, 0, 0, 0] = 6.0
    noisy[:, 1, 0, 0, 0] = 8.0

    mag = complex_magnitude(pred)
    scale = current_magnitude_max(noisy)

    assert tuple(mag.shape) == (2, 1, 16, 8, 8)
    assert tuple(scale.shape) == (2, 1, 1, 1, 1)
    assert torch.allclose(mag, torch.full_like(mag, 5.0))
    assert torch.allclose(scale, torch.full_like(scale, 10.0))


def test_restore_magnitude_uses_per_slice_scales():
    prediction = torch.zeros(1, 2, 3, 4, 4)
    clean = torch.zeros(1, 2, 3, 4, 4)
    prediction[:, 0] = 1.0
    clean[:, 0] = 2.0
    metadata = [{"volume_name": "vol", "slice_scales": [1.0, 2.0, 3.0]}]

    volumes = restore_magnitude_volumes(prediction, clean, metadata)
    name, restored_pred, restored_target = volumes[0]

    assert name == "vol"
    assert restored_pred.shape == (3, 4, 4)
    assert restored_target.shape == (3, 4, 4)
    assert np.allclose(restored_pred[:, 0, 0], np.asarray([1.0, 2.0, 3.0]))
    assert np.allclose(restored_target[:, 0, 0], np.asarray([2.0, 4.0, 6.0]))


def test_volume_metrics_accept_restored_volume_arrays():
    target = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    prediction = target.copy()
    metrics = compute_volume_metrics([("vol", prediction, target)])

    assert metrics["nmse"] == 0.0
    assert np.isfinite(metrics["psnr"]) or np.isinf(metrics["psnr"])


def test_zero_magnitude_loss_weight_skips_sqrt_backward_branch():
    trainer = MulticoilFineTuneTrainer.__new__(MulticoilFineTuneTrainer)
    trainer.loss_fn = nn.L1Loss()
    trainer.config = type(
        "Config",
        (),
        {
            "train": type(
                "Train",
                (),
                {
                    "complex_loss_weight": 1.0,
                    "magnitude_loss_weight": 0.0,
                },
            )()
        },
    )()

    pred = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    clean = torch.zeros_like(pred)
    noisy = torch.zeros(1, 3, 1, 2, 2)

    loss = trainer._loss(pred, clean, noisy)
    assert torch.isfinite(loss)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
