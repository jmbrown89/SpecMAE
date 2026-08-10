from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_options_supports_bool_and_mapping() -> None:
    module = _load_matrix_module()

    assert module._cleanup_options(True) == (True, True)
    assert module._cleanup_options(False) == (False, False)
    assert module._cleanup_options({"run_artifacts": True}) == (True, False)
    assert module._cleanup_options({"result_files": True}) == (False, True)


def test_cleanup_paths_removes_files_and_dirs_under_allowed_root(tmp_path: Path) -> None:
    module = _load_matrix_module()
    output_root = tmp_path / "outputs"
    run_dir = output_root / "runs" / "smoke_run"
    result_file = output_root / "experiments" / "smoke_summary.md"
    run_dir.mkdir(parents=True)
    result_file.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    result_file.write_text("# smoke", encoding="utf-8")

    deleted = module._cleanup_paths([run_dir, result_file], allowed_root=output_root)

    assert deleted == 2
    assert not run_dir.exists()
    assert not result_file.exists()


def test_cleanup_paths_rejects_paths_outside_allowed_root(tmp_path: Path) -> None:
    module = _load_matrix_module()
    output_root = tmp_path / "outputs"
    outside_file = tmp_path / "outside.txt"
    output_root.mkdir()
    outside_file.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="outside output root"):
        module._cleanup_paths([outside_file], allowed_root=output_root)

    assert outside_file.exists()
