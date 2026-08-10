"""Run a curriculum experiment matrix and save aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with 'pip install pyyaml'.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at top level: {path}")
    return loaded


def _to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            args.append(flag if value else f"--no-{key.replace('_', '-')}")
            continue
        args.extend([flag, str(value)])
    return args


def _run_command(cmd: List[str], cwd: Path) -> None:
    joined = " ".join(cmd)
    print(f"\n$ {joined}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _summary_to_row(summary: Dict[str, Any], condition: str, seed: int) -> Dict[str, Any]:
    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss": summary.get("best_val_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "stopped_early": summary.get("stopped_early"),
        "stop_epoch": summary.get("stop_epoch"),
        "restored_best_at_end": summary.get("restored_best_at_end"),
        "run_dir": summary.get("run_dir"),
    }


def _extract_timestamp_from_run_dir(name: str) -> str:
    # Run directories end with a timestamp suffix: YYYYMMDD_HHMMSS.
    parts = name.rsplit("_", 2)
    if len(parts) < 3:
        return ""
    date_part, time_part = parts[-2], parts[-1]
    candidate = f"{date_part}_{time_part}"
    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        return candidate
    return ""


def _find_latest_run_id(artifacts_root_abs: Path, run_prefix: str) -> str | None:
    latest: str | None = None
    for run_dir in artifacts_root_abs.glob(f"{run_prefix}_*"):
        if not run_dir.is_dir():
            continue
        ts = _extract_timestamp_from_run_dir(run_dir.name)
        if not ts:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def _safe_stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    return statistics.stdev(vals)


def _cleanup_options(raw: Any) -> Tuple[bool, bool]:
    """Return cleanup options for run artifacts and result files."""
    if isinstance(raw, bool):
        return raw, raw
    if not isinstance(raw, dict):
        return False, False
    return bool(raw.get("run_artifacts", False)), bool(raw.get("result_files", False))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _cleanup_paths(paths: Iterable[Path], allowed_root: Path) -> int:
    """Delete files or directories, limited to one resolved output root."""
    root = allowed_root.resolve()
    deleted = 0
    for path in paths:
        target = path.resolve()
        if target == root or not _is_relative_to(target, root):
            raise ValueError(f"Refusing to clean path outside output root: {target}")
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted += 1
    return deleted


def run_matrix(
    matrix_path: Path,
    repo_root: Path,
    *,
    resume: bool = False,
    run_id: str | None = None,
) -> None:
    matrix = _load_yaml(matrix_path)

    base_config = str(matrix.get("base_config", "configs/medmnist_2d.yaml"))
    artifacts_root = Path(str(matrix.get("artifacts_root", "outputs/runs")))
    results_root = Path(str(matrix.get("results_root", "outputs/experiments")))
    run_prefix = str(matrix.get("run_prefix", "curriculum_matrix"))
    seeds = list(matrix.get("seeds", [7]))
    shared_overrides = dict(matrix.get("shared_overrides", {}))
    conditions = list(matrix.get("conditions", []))
    cleanup_run_artifacts, cleanup_result_files = _cleanup_options(matrix.get("cleanup_outputs", False))

    if not conditions:
        raise ValueError("No conditions defined in matrix YAML.")

    results_root_abs = repo_root / results_root
    artifacts_root_abs = repo_root / artifacts_root

    results_root_abs.mkdir(parents=True, exist_ok=True)
    artifacts_root_abs.mkdir(parents=True, exist_ok=True)

    selected_run_id = run_id
    if resume and not selected_run_id:
        selected_run_id = _find_latest_run_id(artifacts_root_abs=artifacts_root_abs, run_prefix=run_prefix)
        if selected_run_id:
            print(f"Resume mode enabled. Auto-detected latest run-id: {selected_run_id}")
        else:
            print("Resume mode enabled, but no existing run-id found. Starting a fresh batch.")

    timestamp = selected_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if resume and selected_run_id:
        print("Completed jobs in this batch will be skipped; only missing runs will execute.")

    detailed_rows: List[Dict[str, Any]] = []
    run_dirs_for_cleanup: List[Path] = []

    for condition in conditions:
        name = str(condition["name"])
        cond_overrides = dict(condition.get("overrides", {}))

        for seed in seeds:
            run_name = f"{run_prefix}_{name}_seed{seed}_{timestamp}"
            overrides = dict(shared_overrides)
            overrides.update(cond_overrides)
            overrides["seed"] = seed
            overrides["artifacts_root"] = str(artifacts_root)
            overrides["run_name"] = run_name

            cmd = [
                sys.executable,
                "-m",
                "specmae.training.train",
                "--config",
                base_config,
            ]
            cmd.extend(_to_cli_args(overrides))

            run_dir = repo_root / artifacts_root / run_name
            summary_path = run_dir / "metrics" / "summary.json"
            if cleanup_run_artifacts:
                run_dirs_for_cleanup.append(run_dir)

            if resume:
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    detailed_rows.append(_summary_to_row(summary, condition=name, seed=seed))
                    print(f"[resume] Reusing existing summary for run={run_name}")
                    continue
                if run_dir.exists():
                    # Interrupted runs can leave behind partial folders without summary.json.
                    # Clean them so resume can re-run this job deterministically.
                    print(f"[resume] Found partial run without summary, resetting: {run_dir}")
                    shutil.rmtree(run_dir, ignore_errors=True)

            _run_command(cmd, cwd=repo_root)

            if not summary_path.exists():
                if resume:
                    print(f"[resume] Missing summary after first attempt, retrying once: {run_name}")
                    shutil.rmtree(run_dir, ignore_errors=True)
                    _run_command(cmd, cwd=repo_root)
                if not summary_path.exists():
                    raise FileNotFoundError(
                        f"Missing summary file after run attempt(s): {summary_path}. "
                        "This usually means training was interrupted before final artifact write."
                    )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            detailed_rows.append(_summary_to_row(summary, condition=name, seed=seed))

    output_timestamp = selected_run_id or timestamp
    detailed_csv = results_root_abs / f"{run_prefix}_{output_timestamp}_detailed.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "condition",
            "seed",
            "best_epoch",
            "best_val_loss",
            "final_val_loss",
            "stopped_early",
            "stop_epoch",
            "restored_best_at_end",
            "run_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in detailed_rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for condition, rows in grouped.items():
        best_vals = [float(r["best_val_loss"]) for r in rows]
        final_vals = [float(r["final_val_loss"]) for r in rows]
        aggregate_rows.append(
            {
                "condition": condition,
                "n": len(rows),
                "mean_best_val_loss": _safe_mean(best_vals),
                "std_best_val_loss": _safe_stdev(best_vals),
                "mean_final_val_loss": _safe_mean(final_vals),
                "std_final_val_loss": _safe_stdev(final_vals),
            }
        )

    aggregate_rows.sort(key=lambda r: float(r["mean_best_val_loss"]))

    leaderboard_csv = results_root_abs / f"{run_prefix}_{output_timestamp}_leaderboard.csv"
    with leaderboard_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "condition",
            "n",
            "mean_best_val_loss",
            "std_best_val_loss",
            "mean_final_val_loss",
            "std_final_val_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    report_path = results_root_abs / f"{run_prefix}_{output_timestamp}_summary.md"
    lines = [
        "# Curriculum Matrix Summary",
        "",
        f"- Matrix file: {matrix_path}",
        f"- Detailed CSV: {detailed_csv}",
        f"- Leaderboard CSV: {leaderboard_csv}",
        "",
        "## Leaderboard (mean best val loss)",
        "",
        "| condition | n | mean_best_val_loss | std_best_val_loss | mean_final_val_loss | std_final_val_loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {condition} | {n} | {mean_best_val_loss:.8f} | {std_best_val_loss:.8f} | {mean_final_val_loss:.8f} | {std_final_val_loss:.8f} |".format(
                **row
            )
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    result_files = [detailed_csv, leaderboard_csv, report_path]
    deleted_run_dirs = 0
    deleted_result_files = 0
    if cleanup_run_artifacts:
        deleted_run_dirs = _cleanup_paths(run_dirs_for_cleanup, allowed_root=artifacts_root_abs)
    if cleanup_result_files:
        deleted_result_files = _cleanup_paths(result_files, allowed_root=results_root_abs)

    print("\nMatrix complete.")
    if cleanup_result_files:
        print(f"Cleaned result files: {deleted_result_files}")
    else:
        print(f"Detailed: {detailed_csv}")
        print(f"Leaderboard: {leaderboard_csv}")
        print(f"Summary: {report_path}")
    if cleanup_run_artifacts:
        print(f"Cleaned run artifact directories: {deleted_run_dirs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpecMAE experiment matrix")
    parser.add_argument("--matrix", default="configs/experiments/curriculum_matrix.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed condition/seed runs instead of rerunning them.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional timestamp suffix (YYYYMMDD_HHMMSS) to resume a specific matrix batch.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = (repo_root / args.matrix).resolve()
    run_matrix(
        matrix_path=matrix_path,
        repo_root=repo_root,
        resume=bool(args.resume),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
