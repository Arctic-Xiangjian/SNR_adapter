"""SNRAware model wrapper for native 3D multicoil inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from snraware.projects.mri.denoising.model import DenoisingModel

from .config import BaseModelConfig, CorrectionConfig, PatchShape3D, PreprocessConfig, TrainConfig
from .gmap_adapter import GFactorCorrectionAdapter3D

TARGET_REPLACEMENTS = {
    "ifm.model.config.": "snraware.components.model.config.",
    "ifm.mri.denoising.data.": "snraware.projects.mri.denoising.data.",
}


def _replace_legacy_targets(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_legacy_targets(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_legacy_targets(value) for value in obj]
    if isinstance(obj, str):
        for old, new in TARGET_REPLACEMENTS.items():
            if obj.startswith(old):
                return new + obj[len(old) :]
    return obj


def load_base_model_config(config_path: str | Path, patch_shape: PatchShape3D) -> DictConfig:
    """Load base SNRAware YAML and adapt the 3D cutout shape."""
    raw = OmegaConf.load(config_path)
    fixed = OmegaConf.create(_replace_legacy_targets(OmegaConf.to_container(raw, resolve=False)))
    if not isinstance(fixed, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(fixed).__name__}")
    fixed.dataset.cutout_shape = patch_shape.as_snraware_cutout_hwd()
    return fixed


def _load_raw_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Base model checkpoint does not exist: {path}")

    try:
        scripted = torch.jit.load(str(path), map_location="cpu")
        return {key: value.detach().cpu() for key, value in scripted.state_dict().items()}
    except Exception:
        pass

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")
    tensors = {key: value.detach().cpu() for key, value in payload.items() if torch.is_tensor(value)}
    if not tensors:
        raise ValueError(f"No tensors found in checkpoint: {path}")
    return tensors


def _shape_compatible_state(
    model: nn.Module,
    raw_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    mismatch_keys: list[str] = []
    prefixes = ("", "model.", "base_model.", "module.", "net.")
    for key, value in raw_state.items():
        matched_key = None
        for prefix in prefixes:
            candidate = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            if candidate not in model_state:
                continue
            matched_key = candidate
            if tuple(model_state[candidate].shape) == tuple(value.shape):
                compatible[candidate] = value
            else:
                mismatch_keys.append(
                    f"{candidate}: checkpoint={tuple(value.shape)} model={tuple(model_state[candidate].shape)}"
                )
            break
        if matched_key is None:
            skipped.append(key)
    missing_model_keys = [key for key in model_state if key not in compatible]
    return compatible, skipped, mismatch_keys, missing_model_keys


def build_base_model(
    base_config: BaseModelConfig,
    preprocess: PreprocessConfig,
    train_config: TrainConfig,
) -> tuple[DenoisingModel, DictConfig]:
    """Instantiate SNRAware base model and require strict-compatible weights."""
    patch = train_config.patch
    model_config = load_base_model_config(base_config.config_path, patch)
    model = DenoisingModel(
        config=model_config,
        D=patch.depth,
        H=patch.height,
        W=patch.width,
        C_in=3,
        C_out=2,
    )
    raw_state = _load_raw_state_dict(base_config.checkpoint_path)
    compatible, skipped, mismatch_keys, missing_model_keys = _shape_compatible_state(model, raw_state)
    if mismatch_keys or missing_model_keys:
        examples = (mismatch_keys + [f"missing: {key}" for key in missing_model_keys])[:20]
        raise RuntimeError(
            "Base checkpoint does not fully match the 3D SNRAware model shape. "
            f"matched={len(compatible)} mismatched={len(mismatch_keys)} "
            f"missing_model_keys={len(missing_model_keys)} patch={patch.as_tensor_dhw()} "
            f"mismatched_or_missing_examples={examples}"
        )
    missing, unexpected = model.load_state_dict(compatible, strict=True)
    model.load_report = {
        "matched_keys": len(compatible),
        "mismatched_keys": len(mismatch_keys),
        "total_model_keys": len(model.state_dict()),
        "skipped_tensors": len(skipped),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "patch": {
            "depth": patch.depth,
            "height": patch.height,
            "width": patch.width,
        },
        "model_cutout_shape": patch.as_snraware_cutout_hwd(),
        "eval_crop_size": [int(preprocess.crop_size[0]), int(preprocess.crop_size[1])],
    }
    return model, model_config


class SNRAwareMulticoilWrapper(nn.Module):
    """Frozen SNRAware base plus 3D gmap correction adapter."""

    def __init__(
        self,
        base_model: nn.Module,
        correction_config: CorrectionConfig,
        *,
        patch_shape: PatchShape3D,
    ):
        super().__init__()
        self.base_model = base_model
        self.patch_shape = PatchShape3D.from_value(patch_shape)
        self.gmap_adapter = GFactorCorrectionAdapter3D(correction_config)
        self.use_correction = bool(correction_config.enabled)
        self.last_correction_stats: dict[str, float] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _validate_input(self, x: torch.Tensor) -> None:
        expected = (3, self.patch_shape.depth, self.patch_shape.height, self.patch_shape.width)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B, 3, D, H, W] with tail {expected}, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, *, checkpoint_base_model: bool = False) -> torch.Tensor:
        self._validate_input(x)
        if self.use_correction:
            x = self.gmap_adapter(x)
            self.last_correction_stats = self.gmap_adapter.last_stats
        else:
            self.last_correction_stats = None
        if bool(checkpoint_base_model) and torch.is_grad_enabled():
            y = activation_checkpoint(lambda value: self.base_model(value), x, use_reentrant=False)
        else:
            y = self.base_model(x)
        expected = (2, self.patch_shape.depth, self.patch_shape.height, self.patch_shape.width)
        if y.ndim != 5 or tuple(y.shape[1:]) != expected:
            raise ValueError(f"Expected SNRAware output [B, 2, D, H, W] with tail {expected}, got {tuple(y.shape)}")
        return y


def build_multicoil_model(
    *,
    base_config: BaseModelConfig,
    correction_config: CorrectionConfig,
    preprocess_config: PreprocessConfig,
    train_config: TrainConfig,
) -> tuple[SNRAwareMulticoilWrapper, DictConfig]:
    """Build the wrapped 3D multicoil model."""
    base_model, model_config = build_base_model(base_config, preprocess_config, train_config)
    return (
        SNRAwareMulticoilWrapper(
            base_model,
            correction_config,
            patch_shape=train_config.patch,
        ),
        model_config,
    )
