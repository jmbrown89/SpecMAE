"""Top-level SpecMAE CLI wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_module(module: str, forwarded_args: List[str]) -> int:
    cmd = [sys.executable, "-m", module, *forwarded_args]
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def _run_script(script_path: Path, forwarded_args: List[str]) -> int:
    cmd = [sys.executable, str(script_path), *forwarded_args]
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(prog="specmae", description="SpecMAE command line interface")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("-c", "--config", dest="config", default=None, help="Path to training YAML config")

    subparsers.add_parser("eval", help="Run reconstruction evaluation")

    matrix_parser = subparsers.add_parser("matrix", help="Run experiment matrix")
    matrix_parser.add_argument(
        "-m",
        "--matrix",
        dest="matrix",
        default=None,
        help="Path to experiment matrix YAML",
    )

    args, unknown_args = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "train":
        forwarded: List[str] = []
        if args.config:
            forwarded.extend(["--config", args.config])
        forwarded.extend(unknown_args)
        raise SystemExit(_run_module("specmae.training.train", forwarded))

    if args.command == "eval":
        raise SystemExit(_run_module("specmae.evaluation.reconstruction", list(unknown_args)))

    if args.command == "matrix":
        forwarded = []
        if args.matrix:
            forwarded.extend(["--matrix", args.matrix])
        forwarded.extend(unknown_args)
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "run_experiment_matrix.py"
        raise SystemExit(_run_script(script_path, forwarded))

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
