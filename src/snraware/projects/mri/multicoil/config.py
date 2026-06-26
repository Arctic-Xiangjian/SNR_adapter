"""Typed YAML configuration for the pure multicoil SNRAware project."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _as_path_list(value: str | Path | list[str] | list[Path] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in value]


def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    payload = {} if payload is None else dict(payload)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in payload:
            value = payload[item.name]
        elif item.default is not MISSING:
            value = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = item.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(f"Missing required config field: {cls.__name__}.{item.name}")
        if is_dataclass(item.type) and isinstance(value, dict):
            value = _dataclass_from_dict(item.type, value)
        kwargs[item.name] = value
    return cls(**kwargs)


@dataclass
class H5DataConfig:
    """One split of multicoil H5 data."""

    roots: list[str] = field(default_factory=list)
    format: str = "fastmri"
    kspace_key: str = "kspace"
    target_key: str | None = "reconstruction_rss"
    slice_axis: int = 0
    coil_axis: int = 1
    complex_format: str = "native"
    real_imag_axis: int = -1
    max_slices: int | None = None
    volume_sample_fraction: float | None = None
    volume_sample_seed: int = 42

    def __post_init__(self) -> None:
        self.roots = _as_path_list(self.roots)
        if self.format not in {"fastmri", "generic_h5"}:
            raise ValueError(f"Unsupported data format: {self.format}")
        if self.complex_format not in {"native", "real_imag_last"}:
            raise ValueError(f"Unsupported complex_format: {self.complex_format}")


@dataclass
class SubsetConfig:
    """Training subset policy."""

    mode: str = "none"
    fraction: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        self.mode = self.mode.lower().replace("-", "_")
        if self.mode not in {"none", "random_volume", "random_slice"}:
            raise ValueError(f"Unsupported subset mode: {self.mode}")
        if self.mode != "none":
            if self.fraction is None or not (0.0 < float(self.fraction) <= 1.0):
                raise ValueError("subset.fraction must be in (0, 1] when subset.mode is active")


@dataclass
class PreprocessConfig:
    """Physics preprocessing used by both training and zero-shot inference."""

    crop_size: list[int] = field(default_factory=lambda: [384, 384])
    acc_factor: int = 8
    center_fraction: float = 0.04
    calib_center_fraction: float = 0.04
    sampling_pattern: str = "uniform"
    ncc: int = 8
    grappa_kernel: list[int] = field(default_factory=lambda: [5, 5])
    grappa_lambda: float = 1.0e-4
    cov_corner_fraction: float = 0.125
    cov_shrinkage: float = 0.05
    cov_condition_max: float = 1.0e6
    eig_floor: float = 1.0e-6
    scale_percentile: float = 50.0
    deterministic_mask_from_name: bool = True
    sample_seed: int = 42
    gmap_value: float = 1.0
    cache_dir: str | None = None
    cache_version: str = "multicoil_ones_gmap_v1"

    def __post_init__(self) -> None:
        self.crop_size = [int(v) for v in self.crop_size]
        self.grappa_kernel = [int(v) for v in self.grappa_kernel]
        if len(self.crop_size) != 2:
            raise ValueError("preprocess.crop_size must be [height, width]")
        if len(self.grappa_kernel) != 2:
            raise ValueError("preprocess.grappa_kernel must be [ky, kx]")
        if self.acc_factor <= 0:
            raise ValueError("preprocess.acc_factor must be positive")
        if self.gmap_value <= 0:
            raise ValueError("preprocess.gmap_value must be positive")
        pattern = self.sampling_pattern.lower().replace("-", "_")
        if pattern not in {"uniform", "equispaced", "regular", "random", "random1d"}:
            raise ValueError(f"Unsupported sampling_pattern: {self.sampling_pattern}")
        self.sampling_pattern = pattern


@dataclass
class BaseModelConfig:
    """Frozen SNRAware base model source."""

    variant: str = "large"
    config_path: str = "/working2/arctic/snrawre/SNRAware/checkpoints/large/snraware_large_model.yaml"
    checkpoint_path: str = "/working2/arctic/snrawre/SNRAware/checkpoints/large/snraware_large_model.pts"


@dataclass
class CorrectionConfig:
    """Bounded trainable correction on [real, imag, ones-gmap]."""

    enabled: bool = True
    hidden_chans: int = 64
    gmap_log_bound: float = 1.75
    complex_log_scale_bound: float = 0.75
    gmap_min: float = 0.01
    gmap_max: float = 12.0


@dataclass
class LoraConfig:
    """Minimal LoRA adapter config."""

    enabled: bool = True
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            r"\.attn\.key$",
            r"\.attn\.query$",
            r"\.attn\.value$",
            r"\.attn\.output_proj$",
            r"\.mlp\.0$",
            r"\.mlp\.2$",
        ]
    )


@dataclass
class TrainConfig:
    """Fine-tuning loop controls."""

    mode: str = "warmup_then_both"
    max_epochs: int = 50
    warmup_epochs: int = 4
    gmap_warmup_epochs: int = 2
    batch_size: int = 48
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle_train: bool = True
    train_patch_size: list[int] = field(default_factory=lambda: [64, 64])
    correction_lr: float = 5.0e-4
    adapter_lr: float = 1.0e-4
    weight_decay: float = 0.0
    gradient_clip_val: float = 1.0
    train_pre_post: bool = True
    complex_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    evaluate_every_n_epochs: int = 1
    log_every_n_steps: int = 50
    save_best_only: bool = True
    resume_from: str | None = None
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    run_test_eval: bool = False

    def __post_init__(self) -> None:
        self.train_patch_size = [int(v) for v in self.train_patch_size]
        if self.mode != "warmup_then_both":
            raise ValueError("This clean project keeps only mode='warmup_then_both'")
        if self.gmap_warmup_epochs > self.warmup_epochs:
            raise ValueError("train.gmap_warmup_epochs must be <= train.warmup_epochs")


@dataclass
class RuntimeConfig:
    """Runtime and output controls."""

    device: str = "cuda:0"
    use_bf16: bool = True
    seed: int = 3875032963
    save_root: str = "/working2/arctic/project2/runs"
    run_name: str = "fastmri_x8_cf004_partial05_gmap_ones"


@dataclass
class ProjectConfig:
    """Top-level project config."""

    train_data: H5DataConfig = field(default_factory=H5DataConfig)
    val_data: H5DataConfig = field(default_factory=H5DataConfig)
    test_data: H5DataConfig | None = None
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def from_container(container: dict[str, Any]) -> ProjectConfig:
    """Build a typed config from a YAML/OmegaConf container."""
    return ProjectConfig(
        train_data=_dataclass_from_dict(H5DataConfig, container.get("train_data")),
        val_data=_dataclass_from_dict(H5DataConfig, container.get("val_data")),
        test_data=(
            None
            if container.get("test_data") in (None, "null")
            else _dataclass_from_dict(H5DataConfig, container.get("test_data"))
        ),
        subset=_dataclass_from_dict(SubsetConfig, container.get("subset")),
        preprocess=_dataclass_from_dict(PreprocessConfig, container.get("preprocess")),
        base_model=_dataclass_from_dict(BaseModelConfig, container.get("base_model")),
        correction=_dataclass_from_dict(CorrectionConfig, container.get("correction")),
        lora=_dataclass_from_dict(LoraConfig, container.get("lora")),
        train=_dataclass_from_dict(TrainConfig, container.get("train")),
        runtime=_dataclass_from_dict(RuntimeConfig, container.get("runtime")),
    )


def to_container(config: ProjectConfig) -> dict[str, Any]:
    """Return a plain serializable dict."""
    return asdict(config)


def load_project_config(path: str | Path, overrides: list[str] | None = None) -> ProjectConfig:
    """Load YAML config and optional OmegaConf dotlist overrides."""
    base = OmegaConf.load(path)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    resolved = OmegaConf.to_container(base, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected mapping config, got {type(resolved).__name__}")
    return from_container(resolved)


def save_resolved_config(config: ProjectConfig, path: str | Path) -> None:
    """Write the resolved typed config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(to_container(config)), path)
