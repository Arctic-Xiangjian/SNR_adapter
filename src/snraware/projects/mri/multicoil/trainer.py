"""Explicit 3D multicoil fine-tuning loop."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ProjectConfig, save_resolved_config, to_container
from .lora import (
    apply_lora_to_model,
    count_trainable_parameters,
    has_lora_adapters,
    lora_parameters,
    set_lora_trainable,
)
from .metrics import (
    complex_magnitude,
    compute_volume_metrics,
    current_magnitude_mean,
    restore_magnitude_volumes,
)
from .sliding_window import predict_sliding_window_3d
from .snraware_wrapper import SNRAwareMulticoilWrapper, build_multicoil_model
from .volume_dataset import MulticoilVolumeDataset, collate_multicoil_batch

MULTICOIL_CHECKPOINT_TYPE = "snraware_multicoil_3d_v1"
SHAPE_CONTRACT = "[B,C,D,H,W]"


def _extract_adapter_state(
    model: SNRAwareMulticoilWrapper,
    *,
    include_pre_post: bool,
) -> dict[str, torch.Tensor]:
    if not has_lora_adapters(model.base_model):
        return {}
    selected: dict[str, torch.Tensor] = {}
    for name, tensor in model.base_model.state_dict().items():
        if ".lora_A." in name or ".lora_B." in name or (
            include_pre_post and (name.startswith("pre.") or name.startswith("post."))
        ):
            selected[name] = tensor.detach().cpu()
    return selected


def _is_multicoil_checkpoint(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("checkpoint_type") == MULTICOIL_CHECKPOINT_TYPE


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Create 3D train/val/test dataloaders from typed config."""
    train_dataset = MulticoilVolumeDataset(
        config.train_data,
        config.preprocess,
        split="train",
        subset=config.subset,
        patch_shape=config.train.patch,
    )
    val_loader = None
    if config.val_data.roots:
        val_loader = DataLoader(
            MulticoilVolumeDataset(
                config.val_data,
                config.preprocess,
                split="val",
                patch_shape=config.train.patch,
            ),
            batch_size=int(config.train.val_batch_size),
            shuffle=False,
            num_workers=int(config.train.num_workers),
            pin_memory=bool(config.train.pin_memory),
            persistent_workers=bool(config.train.persistent_workers and config.train.num_workers > 0),
            collate_fn=collate_multicoil_batch,
        )
    test_loader = None
    if config.test_data is not None and config.test_data.roots:
        test_loader = DataLoader(
            MulticoilVolumeDataset(
                config.test_data,
                config.preprocess,
                split="test",
                patch_shape=config.train.patch,
            ),
            batch_size=int(config.train.val_batch_size),
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
    """Trainer for 3D gmap correction plus LoRA fine-tuning."""

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
        self.best_epoch: int | None = None
        self._printed_batch_contract = False

        if config.lora.enabled:
            self.lora_result = apply_lora_to_model(self.model.base_model, config.lora)
        else:
            self.lora_result = None
        if self.lora_result is None or self.lora_result.num_wrapped <= 0:
            raise RuntimeError("LoRA must be enabled and wrap at least one base-model module")
        self._configure_initial_trainability()
        self.optimizer = self._build_optimizer()
        if config.train.resume_from:
            self._load_checkpoint(config.train.resume_from)

    def _configure_initial_trainability(self) -> None:
        _set_module_trainable(self.model.base_model, False)
        _set_module_trainable(self.model.gmap_adapter, True)
        set_lora_trainable(self.model.base_model, False)
        _set_pre_post_trainable(self.model, False)
        self._apply_phase_trainability(0)

    def _adapter_parameters(self) -> list[nn.Parameter]:
        params = lora_parameters(self.model.base_model)
        if bool(self.config.train.train_pre_post):
            params.extend(_pre_post_parameters(self.model))
        return params

    def _build_optimizer(self) -> torch.optim.Optimizer:
        gmap_params = list(self.model.gmap_adapter.parameters())
        adapter_params = self._adapter_parameters()
        groups: list[dict[str, Any]] = [
            {
                "name": "gmap_adapter",
                "params": gmap_params,
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
            _set_module_trainable(self.model.gmap_adapter, False)
            _set_module_trainable(self.model.gmap_adapter.gmap_unet, True)
            self.model.gmap_adapter.log_complex_scale.requires_grad = False
            set_lora_trainable(self.model.base_model, False)
            _set_pre_post_trainable(self.model, False)
        elif phase == "correction_warmup":
            _set_module_trainable(self.model.gmap_adapter, True)
            set_lora_trainable(self.model.base_model, False)
            _set_pre_post_trainable(self.model, False)
        else:
            _set_module_trainable(self.model.gmap_adapter, True)
            set_lora_trainable(self.model.base_model, True)
            _set_pre_post_trainable(self.model, bool(self.config.train.train_pre_post))

        for group in self.optimizer.param_groups if hasattr(self, "optimizer") else []:
            if group.get("name") == "adapter":
                group["lr"] = 0.0 if phase != "joint" else float(self.config.train.adapter_lr)

    def _autocast(self):
        enabled = bool(self.config.runtime.use_bf16 and self.device.type == "cuda")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)

    def _validate_train_batch(self, noisy: torch.Tensor, clean: torch.Tensor) -> None:
        patch = self.config.train.patch
        noisy_tail = (3, patch.depth, patch.height, patch.width)
        clean_tail = (2, patch.depth, patch.height, patch.width)
        if noisy.ndim != 5 or tuple(noisy.shape[1:]) != noisy_tail:
            raise ValueError(f"Expected train noisy [B,3,D,H,W] with tail {noisy_tail}, got {tuple(noisy.shape)}")
        if clean.ndim != 5 or tuple(clean.shape[1:]) != clean_tail:
            raise ValueError(f"Expected train clean [B,2,D,H,W] with tail {clean_tail}, got {tuple(clean.shape)}")

    def _validate_eval_batch(self, noisy: torch.Tensor, clean: torch.Tensor) -> None:
        crop_h, crop_w = (int(v) for v in self.config.preprocess.crop_size)
        if noisy.ndim != 5 or noisy.shape[0] != 1 or noisy.shape[1] != 3:
            raise ValueError(f"Expected eval noisy [1,3,D,H,W], got {tuple(noisy.shape)}")
        if clean.ndim != 5 or clean.shape[0] != 1 or clean.shape[1] != 2:
            raise ValueError(f"Expected eval clean [1,2,D,H,W], got {tuple(clean.shape)}")
        if noisy.shape[-2:] != (crop_h, crop_w) or clean.shape[-2:] != (crop_h, crop_w):
            raise ValueError(
                f"Eval H/W must be preprocess.crop_size {(crop_h, crop_w)}, "
                f"got noisy={tuple(noisy.shape)} clean={tuple(clean.shape)}"
            )
        if noisy.shape[2] != clean.shape[2]:
            raise ValueError(f"Eval noisy/clean D mismatch: noisy={tuple(noisy.shape)} clean={tuple(clean.shape)}")

    def _loss(self, pred: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 5 or clean.ndim != 5 or noisy.ndim != 5:
            raise ValueError("3D multicoil loss accepts only [B,C,D,H,W] tensors")
        scale = current_magnitude_mean(noisy).to(device=pred.device, dtype=pred.dtype)
        complex_loss = self.loss_fn(pred / scale, clean / scale)
        magnitude_loss = self.loss_fn(complex_magnitude(pred) / scale, complex_magnitude(clean) / scale)
        return (
            float(self.config.train.complex_loss_weight) * complex_loss
            + float(self.config.train.magnitude_loss_weight) * magnitude_loss
        )

    def _checkpoint_base_model_for_epoch(self, epoch: int) -> bool:
        return bool(self.config.train.gradient_checkpoint_frozen_base and self.model.training)

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
            "lr_gmap_adapter",
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

    def _print_batch_contract(self, noisy: torch.Tensor, clean: torch.Tensor, pred: torch.Tensor) -> None:
        if self._printed_batch_contract:
            return
        print(
            "first train batch tensor contract [B,C,D,H,W]: "
            f"noisy={tuple(noisy.shape)} clean={tuple(clean.shape)} pred={tuple(pred.shape)}",
            flush=True,
        )
        self._printed_batch_contract = True

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
            noisy = batch["noisy"].to(self.device, non_blocking=True).float()
            clean = batch["clean"].to(self.device, non_blocking=True).float()
            self._validate_train_batch(noisy, clean)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(noisy, checkpoint_base_model=self._checkpoint_base_model_for_epoch(epoch))
                loss = self._loss(pred.float(), clean.float(), noisy.float())
            self._print_batch_contract(noisy, clean, pred)
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
        stats = getattr(self.model.gmap_adapter, "last_stats", None) or {}
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
                "lr_gmap_adapter": self._group_lr("gmap_adapter"),
                "lr_adapter": self._group_lr("adapter"),
                "trainable_parameters": count_trainable_parameters(self.model),
                **result,
            }
        )
        return result

    def _predict_for_eval(self, noisy: torch.Tensor) -> torch.Tensor:
        patch = self.config.train.patch
        if tuple(noisy.shape[2:]) == patch.as_tensor_dhw():
            return self.model(noisy, checkpoint_base_model=False)
        return predict_sliding_window_3d(
            self.model,
            noisy,
            patch=patch,
            overlap=self.config.train.inference_overlap,
            patch_batch_size=int(self.config.train.eval_patch_batch_size),
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None, *, stage: str, epoch: int | None) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        losses: list[float] = []
        volumes: list[tuple[str, np.ndarray, np.ndarray]] = []
        limit = self.config.train.limit_val_batches
        for step, batch in enumerate(loader):
            if limit is not None and step >= int(limit):
                break
            noisy = batch["noisy"].to(self.device, non_blocking=True).float()
            clean = batch["clean"].to(self.device, non_blocking=True).float()
            self._validate_eval_batch(noisy, clean)
            pred = self._predict_for_eval(noisy).float()
            losses.append(float(self._loss(pred, clean, noisy).cpu().item()))
            volumes.extend(restore_magnitude_volumes(pred, clean, batch["metadata"]))
        result = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            **compute_volume_metrics(volumes),
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

    def _save_checkpoint(self, path: Path, *, epoch: int, metrics: dict[str, float]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lora_adapter = _extract_adapter_state(
            self.model,
            include_pre_post=bool(self.config.train.train_pre_post),
        )
        if not lora_adapter:
            raise RuntimeError("No LoRA/pre-post adapter tensors were selected for checkpointing")
        patch = self.config.train.patch
        payload = {
            "checkpoint_type": MULTICOIL_CHECKPOINT_TYPE,
            "shape_contract": SHAPE_CONTRACT,
            "patch": {"depth": patch.depth, "height": patch.height, "width": patch.width},
            "epoch": int(epoch),
            "metrics": metrics or {},
            "best_val_psnr": (
                None if not np.isfinite(self.best_val_psnr) else float(self.best_val_psnr)
            ),
            "best_epoch": None if self.best_epoch is None else int(self.best_epoch),
            "config": to_container(self.config),
            "gmap_adapter": {
                key: value.detach().cpu()
                for key, value in self.model.gmap_adapter.state_dict().items()
            },
            "lora_adapter": lora_adapter,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not _is_multicoil_checkpoint(payload):
            raise ValueError(
                "This is a 3D-only pipeline. 2D adapter checkpoints are not supported. "
                f"Unsupported checkpoint: {checkpoint_path}"
            )
        self.model.gmap_adapter.load_state_dict(payload["gmap_adapter"], strict=True)
        lora_state = payload.get("lora_adapter", {})
        if not lora_state:
            raise RuntimeError(f"Checkpoint is missing LoRA/pre-post adapter tensors: {checkpoint_path}")
        if not has_lora_adapters(self.model.base_model):
            apply_lora_to_model(self.model.base_model, self.config.lora)
        expected_lora_state = _extract_adapter_state(
            self.model,
            include_pre_post=bool(self.config.train.train_pre_post),
        )
        expected_keys = set(expected_lora_state)
        saved_keys = set(lora_state)
        missing_adapter_keys = sorted(expected_keys - saved_keys)
        unexpected_adapter_keys = sorted(saved_keys - expected_keys)
        if missing_adapter_keys or unexpected_adapter_keys:
            raise RuntimeError(
                "Adapter checkpoint keys do not match the current 3D model/config: "
                f"missing={missing_adapter_keys[:20]} unexpected={unexpected_adapter_keys[:20]}"
            )
        missing, unexpected = self.model.base_model.load_state_dict(lora_state, strict=False)
        missing_loaded_keys = sorted(set(missing) & expected_keys)
        if missing_loaded_keys or unexpected:
            raise RuntimeError(
                "Failed to load multicoil 3D adapter checkpoint: "
                f"missing={missing_loaded_keys[:20]} unexpected={unexpected[:20]}"
            )
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        best_val_psnr = payload.get("best_val_psnr")
        best_epoch = payload.get("best_epoch")
        if best_val_psnr is not None:
            self.best_val_psnr = float(best_val_psnr)
            self.best_epoch = None if best_epoch is None else int(best_epoch)
        else:
            checkpoint_psnr = payload.get("metrics", {}).get("psnr") if isinstance(payload.get("metrics"), dict) else None
            if checkpoint_psnr is not None and np.isfinite(float(checkpoint_psnr)):
                self.best_val_psnr = float(checkpoint_psnr)
                self.best_epoch = int(payload.get("epoch", -1))
        self.current_epoch = int(payload.get("epoch", -1)) + 1

    def fit(self) -> dict[str, Any]:
        save_resolved_config(self.config, self.run_dir / "config_resolved.yaml")
        last_ckpt_path = self.run_dir / "last.pth"
        best_ckpt_path = self.run_dir / "best_psnr.pth"
        save_best_only = bool(self.config.train.save_best_only)
        if save_best_only and self.val_loader is None:
            raise ValueError("train.save_best_only=true requires a validation loader")
        if not self.config.train.resume_from:
            for stale_path in (last_ckpt_path, best_ckpt_path):
                if stale_path.exists():
                    stale_path.unlink()
        summary = {
            "run_dir": str(self.run_dir),
            "best_val_psnr": None if not np.isfinite(self.best_val_psnr) else float(self.best_val_psnr),
            "best_epoch": self.best_epoch,
            "best_checkpoint": (
                str(best_ckpt_path)
                if save_best_only and self.best_epoch is not None and best_ckpt_path.exists()
                else None
            ),
            "last_checkpoint": None if save_best_only else str(last_ckpt_path),
            "save_best_only": save_best_only,
            "shape_contract": SHAPE_CONTRACT,
            "lora_wrapped": None if self.lora_result is None else self.lora_result.wrapped_names,
        }
        for epoch in range(self.current_epoch, int(self.config.train.max_epochs)):
            self.train_epoch(epoch)
            val_metrics: dict[str, float] = {}
            if self.val_loader is not None and (epoch + 1) % int(self.config.train.evaluate_every_n_epochs) == 0:
                val_metrics = self.evaluate(self.val_loader, stage="val", epoch=epoch)
                val_psnr = float(val_metrics.get("psnr", float("-inf")))
                if val_psnr > self.best_val_psnr:
                    self.best_val_psnr = val_psnr
                    self.best_epoch = int(epoch)
                    if save_best_only:
                        self._save_checkpoint(best_ckpt_path, epoch=epoch, metrics=val_metrics)
                    summary["best_val_psnr"] = val_psnr
                    summary["best_epoch"] = self.best_epoch
                    summary["best_checkpoint"] = str(best_ckpt_path) if save_best_only else None
            if not save_best_only:
                self._save_checkpoint(last_ckpt_path, epoch=epoch, metrics=val_metrics)
            with (self.run_dir / "summary.json").open("w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        if self.config.train.run_test_eval and self.test_loader is not None:
            summary["test"] = self.evaluate(self.test_loader, stage="test", epoch=None)
            with (self.run_dir / "summary.json").open("w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        return summary


def _print_startup_contract(config: ProjectConfig) -> None:
    patch = config.train.patch
    overlap = config.train.inference_overlap
    crop_h, crop_w = (int(v) for v in config.preprocess.crop_size)
    print(
        "tensor contract: [B,C,D,H,W]; "
        f"train noisy=[B,3,{patch.depth},{patch.height},{patch.width}] "
        f"clean=[B,2,{patch.depth},{patch.height},{patch.width}]; "
        f"eval noisy=[1,3,D,{crop_h},{crop_w}]; "
        f"patch D/H/W={patch.as_tensor_dhw()} overlap D/H/W={overlap.as_tensor_dhw()}",
        flush=True,
    )


def run_training(config: ProjectConfig) -> dict[str, Any]:
    """Build dataloaders/model and run 3D fine-tuning."""
    seed_everything(int(config.runtime.seed))
    _print_startup_contract(config)
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
