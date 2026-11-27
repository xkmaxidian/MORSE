"""Command-line entry; dataset-specific settings live in JSON, not in source."""
import argparse
from pathlib import Path
from .config import read_config
from .pipeline import run_pipeline


def main(dataset, project_root):
    root = Path(project_root).resolve()
    parser = argparse.ArgumentParser(description=f"MORSE full-model workflow: {dataset}")
    parser.add_argument("--config", type=Path, default=root / "configs" / f"{dataset}.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="cpu or a PyTorch CUDA device such as cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if config["dataset"] != dataset:
        parser.error("Configuration dataset does not match this entry point.")
    for key in ("device", "epochs", "seed"):
        value = getattr(args, key)
        if value is not None:
            config["model"][key] = value
    run_pipeline(config, root, args.output_dir)
