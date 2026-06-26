"""Run zero-shot or fine-tuned SNRAware inference on custom multicoil H5 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from snraware.projects.mri.multicoil.adapter import apply_lora_to_model
from snraware.projects.mri.multicoil.config import load_project_config
from snraware.projects.mri.multicoil.h5_dataset import MulticoilH5Dataset, collate_multicoil_batch
from snraware.projects.mri.multicoil.snraware_wrapper import build_multicoil_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Multicoil YAML config.")
    parser.add_argument("--input-root", required=True, help="Custom H5 file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for prediction npz files.")
    parser.add_argument("--checkpoint", default=None, help="Optional project2 fine-tuned checkpoint.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument("--max-slices", type=int, default=None, help="Optional cap for quick inference.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def _load_optional_checkpoint(model: torch.nn.Module, checkpoint_path: str | None) -> None:
    if checkpoint_path in (None, "", "null"):
        return
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_project_config(args.config, overrides=args.overrides)
    data_config = config.val_data
    data_config.roots = [args.input_root]
    data_config.target_key = None
    data_config.max_slices = args.max_slices
    device_name = args.device or config.runtime.device
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    dataset = MulticoilH5Dataset(
        data_config,
        config.preprocess,
        split="zero_shot",
        subset=None,
        train_patch_size=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_multicoil_batch,
    )

    model, _model_config = build_multicoil_model(
        base_config=config.base_model,
        correction_config=config.correction,
        preprocess_config=config.preprocess,
    )
    if config.lora.enabled:
        apply_lora_to_model(model.base_model, config.lora)
    _load_optional_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for batch in loader:
        pred = model(batch["noisy"].to(device)).detach().cpu().numpy()
        for item_idx, metadata in enumerate(batch["metadata"]):
            name = f"{Path(str(metadata['volume_name'])).stem}_slice_{int(metadata['slice_idx']):04d}.npz"
            out_path = output_dir / name
            np.savez_compressed(
                out_path,
                pred_real=pred[item_idx, 0].astype(np.float32),
                pred_imag=pred[item_idx, 1].astype(np.float32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
            )
            manifest.append({"path": str(out_path), "metadata": metadata})
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
