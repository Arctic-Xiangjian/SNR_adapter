"""Train the pure multicoil SNRAware fine-tuning project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from snraware.projects.mri.multicoil.config import load_project_config, to_container
from snraware.projects.mri.multicoil.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to multicoil YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. runtime.device=cuda:0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without touching data or starting training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config, overrides=args.overrides)
    resolved = to_container(config)
    if args.dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True))
        return
    summary = run_training(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
