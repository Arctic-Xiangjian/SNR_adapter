"""Explicit multicoil fine-tuning loop."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .adapter import (
    apply_lora_to_model,
    count_trainable_parameters,
    lora_parameters,
    set_lora_trainable,
)
from .config import ProjectConfig, save_resolved_config, to_container
from .h5_dataset import MulticoilH5Dataset, collate_multicoil_batch
from .snraware_wrapper import SNRAwareMulticoilWrapper, build_multicoil_model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, 2, H, W] complex tensor to [B, 1, H, W] magnitude."""
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, H, W], got {tuple(x.shape)}")
    return torch.sqrt(x[:, 0:1].square() + x[:, 1:2].square())


def simple_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute a simple batch PSNR in magnitude domain."""
    pred_mag = complex_magnitude(pred).detach().float()
    target_mag = complex_magnitude(target).detach().float()
    mse = torch.mean((pred_mag - target_mag) ** 2)
    peak = torch.amax(target_mag).clamp_min(1.0e-6)
    return float((20.0 * torch.log10(peak) - 10.0 * torch.log10(mse.clamp_min(1.0e-12))).item())


def simple_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute NMSE in magnitude domain."""
    pred_mag = complex_magnitude(pred).detach().float()
    target_mag = complex_magnitude(target).detach().float()
    numerator = torch.sum((pred_mag - target_mag) ** 2)
    denominator = torch.sum(target_mag**2).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def fastmri_current_magnitude_mean(noisy: torch.Tensor) -> torch.Tensor:
    """Per-sample scale used by the old FastMRI fine-tune loss."""
    if noisy.ndim != 4 or noisy.shape[1] < 2:
        raise ValueError(f"Expected [B, C>=2, H, W], got {tuple(noisy.shape)}")
    magnitude = torch.sqrt(noisy[:, 0:1].square() + noisy[:, 1:2].square())
    scale = magnitude.mean(dim=(-2, -1), keepdim=True)
    if not torch.isfinite(scale).all():
        raise ValueError("Current-mean loss normalization received non-finite scale")
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def _metric_nmse(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.sum((prediction - target) ** 2))
    denominator = float(np.sum(target**2))
    return numerator / max(denominator, 1.0e-12)


def _metric_psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    peak = float(np.max(target))
    return 20.0 * np.log10(max(peak, 1.0e-6)) - 10.0 * np.log10(max(mse, 1.0e-12))


def _metric_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        structural_similarity = None

    if prediction.ndim == 3:
        values = [_metric_ssim(prediction[index], target[index]) for index in range(prediction.shape[0])]
        return float(np.mean(values)) if values else float("nan")
    data_range = float(np.max(target) - np.min(target))
    if data_range <= 0:
        return float("nan")
    if structural_similarity is not None:
        return float(structural_similarity(target, prediction, data_range=data_range))
    pred64 = prediction.astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(pred64.mean())
    mu_y = float(target64.mean())
    var_x = float(pred64.var())
    var_y = float(target64.var())
    cov_xy = float(((pred64 - mu_x) * (target64 - mu_y)).mean())
    return ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )


def _group_slices_into_volumes(
    volume_names: list[str],
    slice_indices: list[int],
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for volume_name, slice_idx, prediction, target in zip(volume_names, slice_indices, predictions, targets):
        grouped[str(volume_name)].append((int(slice_idx), prediction, target))
    volumes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for volume_name, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        volumes[volume_name] = (
            np.stack([entry[1] for entry in entries], axis=0),
            np.stack([entry[2] for entry in entries], axis=0),
        )
    return volumes


def _compute_volume_metrics(grouped: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    if not grouped:
        return {"psnr": float("nan"), "ssim": float("nan"), "nmse": float("nan")}
    psnrs: list[float] = []
    ssims: list[float] = []
    nmses: list[float] = []
    for prediction, target in grouped.values():
        psnrs.append(_metric_psnr(prediction, target))
        ssim_value = _metric_ssim(prediction, target)
        if np.isfinite(ssim_value):
            ssims.append(float(ssim_value))
        nmses.append(_metric_nmse(prediction, target))
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "nmse": float(np.mean(nmses)) if nmses else float("nan"),
    }


def _set_module_trainable(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(flag)


def _pre_post_parameters(model: SNRAwareMulticoilWrapper) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module_name in ("pre", "post"):
        module = getattr(model.base_model, module_name, None)
        if module is not None:
            params.extend(module.parameters())
    return params


def _set_pre_post_trainable(model: SNRAwareMulticoilWrapper, flag: bool) -> None:
    for parameter in _pre_post_parameters(model):
        parameter.requires_grad = bool(flag)


def build_dataloaders(config: ProjectConfig) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Create train/val/test dataloaders from typed config."""
    train_dataset = MulticoilH5Dataset(
        config.train_data,
        config.preprocess,
        split="train",
        subset=config.subset,
        train_patch_size=config.train.train_patch_size,
    )
    val_loader = None
    if config.val_data.roots:
        val_loader = DataLoader(
            MulticoilH5Dataset(config.val_data, config.preprocess, split="val"),
            batch_size=max(1, min(8, int(config.train.batch_size))),
            shuffle=False,
            num_workers=int(config.train.num_workers),
            pin_memory=bool(config.train.pin_memory),
            persistent_workers=bool(config.train.persistent_workers and config.train.num_workers > 0),
            collate_fn=collate_multicoil_batch,
        )
    test_loader = None
    if config.test_data is not None and config.test_data.roots:
        test_loader = DataLoader(
            MulticoilH5Dataset(config.test_data, config.preprocess, split="test"),
            batch_size=max(1, min(8, int(config.train.batch_size))),
            shuffle=False,
            num_workers=int(config.train.num_workers),
            pin_memory=bool(config.train.pin_memory),
            persistent_workers=bool(config.train.persistent_workers and config.train.num_workers > 0),
            collate_fn=collate_multicoil_batch,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=bool(config.train.shuffle_train),
        num_workers=int(config.train.num_workers),
        pin_memory=bool(config.train.pin_memory),
        persistent_workers=bool(config.train.persistent_workers and config.train.num_workers > 0),
        collate_fn=collate_multicoil_batch,
    )
    return train_loader, val_loader, test_loader


class MulticoilFineTuneTrainer:
    """Small trainer for ones-gmap correction plus LoRA fine-tuning."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        model: SNRAwareMulticoilWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        run_dir: str | Path,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = nn.L1Loss()
        self.current_epoch = 0
        self.best_val_psnr = float("-inf")

        if config.lora.enabled:
            self.lora_result = apply_lora_to_model(self.model.base_model, config.lora)
        else:
            self.lora_result = None
        self._configure_initial_trainability()
        self.optimizer = self._build_optimizer()
        if config.train.resume_from:
            self._load_checkpoint(config.train.resume_from)

    def _configure_initial_trainability(self) -> None:
        _set_module_trainable(self.model.base_model, False)
        _set_module_trainable(self.model.correction_adapter, True)
        set_lora_trainable(self.model.base_model, False)
        _set_pre_post_trainable(self.model, False)
        self._apply_phase_trainability(0)

    def _adapter_parameters(self) -> list[nn.Parameter]:
        params = lora_parameters(self.model.base_model)
        if bool(self.config.train.train_pre_post):
            params.extend(_pre_post_parameters(self.model))
        return params

    def _build_optimizer(self) -> torch.optim.Optimizer:
        correction_params = list(self.model.correction_adapter.parameters())
        adapter_params = self._adapter_parameters()
        groups: list[dict[str, Any]] = [
            {
                "name": "physics_correction",
                "params": correction_params,
                "lr": float(self.config.train.correction_lr),
                "weight_decay": float(self.config.train.weight_decay),
            }
        ]
        if adapter_params:
            groups.append(
                {
                    "name": "adapter",
                    "params": adapter_params,
                    "lr": 0.0,
                    "weight_decay": float(self.config.train.weight_decay),
                }
            )
        return torch.optim.AdamW(groups)

    def _phase_name(self, epoch: int) -> str:
        if epoch < int(self.config.train.gmap_warmup_epochs):
            return "gmap_warmup"
        if epoch < int(self.config.train.warmup_epochs):
            return "correction_warmup"
        return "joint"

    def _apply_phase_trainability(self, epoch: int) -> None:
        phase = self._phase_name(epoch)
        if phase == "gmap_warmup":
            _set_module_trainable(self.model.correction_adapter, False)
            _set_module_trainable(self.model.correction_adapter.gmap_net, True)
            self.model.correction_adapter.log_complex_scale.requires_grad = False
            set_lora_trainable(self.model.base_model, False)
            _set_pre_post_trainable(self.model, False)
        elif phase == "correction_warmup":
            _set_module_trainable(self.model.correction_adapter, True)
            set_lora_trainable(self.model.base_model, False)
            _set_pre_post_trainable(self.model, False)
        else:
            _set_module_trainable(self.model.correction_adapter, True)
            set_lora_trainable(self.model.base_model, True)
            _set_pre_post_trainable(self.model, bool(self.config.train.train_pre_post))

        for group in self.optimizer.param_groups if hasattr(self, "optimizer") else []:
            if group.get("name") == "adapter":
                group["lr"] = 0.0 if phase != "joint" else float(self.config.train.adapter_lr)

    def _autocast(self):
        enabled = bool(self.config.runtime.use_bf16 and self.device.type == "cuda")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)

    def _loss(self, pred: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        scale = fastmri_current_magnitude_mean(noisy).to(device=pred.device, dtype=pred.dtype)
        complex_loss = self.loss_fn(pred / scale, clean / scale)
        magnitude_loss = self.loss_fn(complex_magnitude(pred) / scale, complex_magnitude(clean) / scale)
        return (
            float(self.config.train.complex_loss_weight) * complex_loss
            + float(self.config.train.magnitude_loss_weight) * magnitude_loss
        )

    def _checkpoint_base_model_for_epoch(self, epoch: int) -> bool:
        if not bool(self.config.train.gradient_checkpoint_frozen_base):
            return False
        return self._phase_name(epoch) != "joint"

    def _keep_frozen_base_eval_for_epoch(self, epoch: int) -> bool:
        if not bool(self.config.train.frozen_base_eval):
            return False
        return self._phase_name(epoch) != "joint"

    def _write_metrics_row(self, row: dict[str, Any]) -> None:
        path = self.run_dir / "metrics.csv"
        fields = [
            "stage",
            "epoch",
            "phase",
            "loss",
            "psnr",
            "ssim",
            "nmse",
            "lr_physics_correction",
            "lr_adapter",
            "trainable_parameters",
            "complex_scale",
            "gmap_mean",
            "gmap_p95",
            "gmap_max",
        ]
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fields})

    def _group_lr(self, name: str) -> float | None:
        for group in self.optimizer.param_groups:
            if group.get("name") == name:
                return float(group["lr"])
        return None

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self._apply_phase_trainability(epoch)
        if self._keep_frozen_base_eval_for_epoch(epoch):
            self.model.base_model.eval()
        losses: list[float] = []
        limit = self.config.train.limit_train_batches
        for step, batch in enumerate(self.train_loader):
            if limit is not None and step >= int(limit):
                break
            noisy = batch["noisy"].to(self.device, non_blocking=True)
            clean = batch["clean"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(noisy, checkpoint_base_model=self._checkpoint_base_model_for_epoch(epoch))
                loss = self._loss(pred.float(), clean.float(), noisy.float())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.train.gradient_clip_val))
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if (step + 1) % int(self.config.train.log_every_n_steps) == 0:
                print(
                    f"epoch={epoch} step={step + 1} phase={self._phase_name(epoch)} "
                    f"loss={np.mean(losses):.6f}",
                    flush=True,
                )
        stats = getattr(self.model.correction_adapter, "last_stats", None) or {}
        result = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "complex_scale": float(stats.get("complex_scale", float("nan"))),
            "gmap_mean": float(stats.get("gmap_mean", float("nan"))),
            "gmap_p95": float(stats.get("gmap_p95", float("nan"))),
            "gmap_max": float(stats.get("gmap_max", float("nan"))),
        }
        self._write_metrics_row(
            {
                "stage": "train",
                "epoch": epoch,
                "phase": self._phase_name(epoch),
                "lr_physics_correction": self._group_lr("physics_correction"),
                "lr_adapter": self._group_lr("adapter"),
                "trainable_parameters": count_trainable_parameters(self.model),
                **result,
            }
        )
        return result

    def _eval_patch_positions(self, size: int, patch: int, overlap: int) -> list[int]:
        if patch >= size:
            return [0]
        step = max(1, patch - overlap)
        positions = list(range(0, size - patch + 1, step))
        final = size - patch
        if positions[-1] != final:
            positions.append(final)
        return positions

    def _should_use_eval_patch_inference(self, noisy: torch.Tensor) -> bool:
        patch_h, patch_w = (int(v) for v in self.config.train.train_patch_size)
        return noisy.shape[-2] != patch_h or noisy.shape[-1] != patch_w

    def _predict_direct_eval(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.model(noisy, checkpoint_base_model=False)

    def _predict_sliding_window_eval(self, noisy: torch.Tensor) -> torch.Tensor:
        batch_size, _channels, height, width = noisy.shape
        patch_h, patch_w = (int(v) for v in self.config.train.train_patch_size)
        overlap_h, overlap_w = int(self.config.train.overlap_for_inference[0]), int(
            self.config.train.overlap_for_inference[1]
        )
        if patch_h > height or patch_w > width:
            raise ValueError(f"Eval patch {(patch_h, patch_w)} must fit inside noisy image {(height, width)}")
        tops = self._eval_patch_positions(height, patch_h, overlap_h)
        lefts = self._eval_patch_positions(width, patch_w, overlap_w)
        prediction_sum = torch.zeros(batch_size, 2, height, width, device=noisy.device, dtype=torch.float32)
        weight_sum = torch.zeros(batch_size, 1, height, width, device=noisy.device, dtype=torch.float32)
        pending: list[torch.Tensor] = []
        coords: list[tuple[int, int]] = []
        patch_budget = max(1, int(self.config.train.eval_patch_batch_size))

        def flush() -> None:
            if not pending:
                return
            chunk = torch.cat(pending, dim=0)
            output = self._predict_direct_eval(chunk).float()
            cursor = 0
            for top, left in coords:
                patch_output = output[cursor : cursor + batch_size]
                cursor += batch_size
                prediction_sum[..., top : top + patch_h, left : left + patch_w] += patch_output
                weight_sum[..., top : top + patch_h, left : left + patch_w] += 1.0
            pending.clear()
            coords.clear()

        for top in tops:
            for left in lefts:
                pending.append(noisy[..., top : top + patch_h, left : left + patch_w])
                coords.append((top, left))
                if len(pending) * batch_size >= patch_budget:
                    flush()
        flush()
        return prediction_sum / weight_sum.clamp_min(1.0)

    def _predict_for_eval(self, noisy: torch.Tensor) -> torch.Tensor:
        if self._should_use_eval_patch_inference(noisy):
            return self._predict_sliding_window_eval(noisy)
        return self._predict_direct_eval(noisy)

    def _metadata_to_numpy(
        self,
        prediction: torch.Tensor,
        clean: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> tuple[list[str], list[int], list[np.ndarray], list[np.ndarray]]:
        pred_mag = complex_magnitude(prediction).detach().cpu().float().numpy()
        target_mag = complex_magnitude(clean).detach().cpu().float().numpy()
        volume_names: list[str] = []
        slice_indices: list[int] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index, entry in enumerate(metadata):
            mean = float(entry.get("mean", 0.0))
            std = float(entry.get("std", entry.get("scale", 1.0)))
            volume_names.append(str(entry.get("volume_name", entry.get("name", "unknown_volume"))))
            slice_indices.append(int(entry.get("slice_idx", index)))
            predictions.append((pred_mag[index, 0] * std + mean).astype(np.float32, copy=False))
            targets.append((target_mag[index, 0] * std + mean).astype(np.float32, copy=False))
        return volume_names, slice_indices, predictions, targets

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None, *, stage: str, epoch: int | None) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        losses: list[float] = []
        volume_names: list[str] = []
        slice_indices: list[int] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        limit = self.config.train.limit_val_batches
        for step, batch in enumerate(loader):
            if limit is not None and step >= int(limit):
                break
            noisy = batch["noisy"].to(self.device, non_blocking=True).float()
            clean = batch["clean"].to(self.device, non_blocking=True).float()
            pred = self._predict_for_eval(noisy).float()
            losses.append(float(self._loss(pred, clean, noisy).cpu().item()))
            batch_volume_names, batch_slice_indices, batch_predictions, batch_targets = self._metadata_to_numpy(
                pred,
                clean,
                batch["metadata"],
            )
            volume_names.extend(batch_volume_names)
            slice_indices.extend(batch_slice_indices)
            predictions.extend(batch_predictions)
            targets.extend(batch_targets)
        grouped = _group_slices_into_volumes(volume_names, slice_indices, predictions, targets)
        result = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            **_compute_volume_metrics(grouped),
        }
        self._write_metrics_row(
            {
                "stage": stage,
                "epoch": "" if epoch is None else epoch,
                "phase": "" if epoch is None else self._phase_name(epoch),
                **result,
            }
        )
        return result

    def _save_checkpoint(self, path: Path, *, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "checkpoint_type": "snraware_project2_multicoil_v1",
                "epoch": int(epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": to_container(self.config),
                "metrics": metrics,
            },
            path,
        )

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(payload["model_state_dict"], strict=False)
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.current_epoch = int(payload.get("epoch", -1)) + 1

    def fit(self) -> dict[str, Any]:
        save_resolved_config(self.config, self.run_dir / "config_resolved.yaml")
        summary = {
            "run_dir": str(self.run_dir),
            "best_val_psnr": None,
            "best_checkpoint": None,
            "lora_wrapped": None if self.lora_result is None else self.lora_result.wrapped_names,
        }
        for epoch in range(self.current_epoch, int(self.config.train.max_epochs)):
            train_metrics = self.train_epoch(epoch)
            val_metrics: dict[str, float] = {}
            if self.val_loader is not None and (epoch + 1) % int(self.config.train.evaluate_every_n_epochs) == 0:
                val_metrics = self.evaluate(self.val_loader, stage="val", epoch=epoch)
                val_psnr = float(val_metrics.get("psnr", float("-inf")))
                if val_psnr > self.best_val_psnr:
                    self.best_val_psnr = val_psnr
                    self._save_checkpoint(self.run_dir / "best_psnr.pth", epoch=epoch, metrics=val_metrics)
                    summary["best_val_psnr"] = val_psnr
                    summary["best_checkpoint"] = str(self.run_dir / "best_psnr.pth")
            if not bool(self.config.train.save_best_only):
                self._save_checkpoint(self.run_dir / f"epoch_{epoch:04d}.pth", epoch=epoch, metrics=train_metrics)
            with (self.run_dir / "summary.json").open("w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        if self.config.train.run_test_eval and self.test_loader is not None:
            summary["test"] = self.evaluate(self.test_loader, stage="test", epoch=None)
            with (self.run_dir / "summary.json").open("w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        return summary


def run_training(config: ProjectConfig) -> dict[str, Any]:
    """Build dataloaders/model and run fine-tuning."""
    seed_everything(int(config.runtime.seed))
    run_dir = Path(config.runtime.save_root) / str(config.runtime.run_name)
    train_loader, val_loader, test_loader = build_dataloaders(config)
    model, _model_config = build_multicoil_model(
        base_config=config.base_model,
        correction_config=config.correction,
        preprocess_config=config.preprocess,
        train_config=config.train,
    )
    trainer = MulticoilFineTuneTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_dir=run_dir,
    )
    return trainer.fit()
