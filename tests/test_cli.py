"""Tests for the unified CLI (metagate command)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from metagate_hipporag.cli import (
    _DATASETS,
    _METHODS,
    _build_parser,
    app,
    main,
)


# ── Parser construction ───────────────────────────────────────────────────────


def test_parser_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """No args prints help and returns code 1."""
    rc = main([])
    assert rc == 1
    captured = capsys.readouterr().out
    assert "MetaGate-HippoRAG 2 experiment CLI" in captured


def test_parser_help_flag() -> None:
    """--help raises SystemExit(0)."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


# ── prepare-data ──────────────────────────────────────────────────────────────


def test_prepare_data_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["prepare-data", "--help"])
    assert exc.value.code == 0


@patch("metagate_hipporag.cli._cmd_prepare_data")
def test_prepare_data_dispatches(mock_fn: object) -> None:
    parser = _build_parser()
    args = parser.parse_args(["prepare-data", "--config", "configs/experiment.yaml"])
    assert hasattr(args, "func")
    args.func(args)
    mock_fn.assert_called_once_with(args)  # type: ignore[union-attr]


# ── prepare-openie ────────────────────────────────────────────────────────────


def test_prepare_openie_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["prepare-openie", "--help"])
    assert exc.value.code == 0


def test_prepare_openie_requires_dataset_phase_workspace() -> None:
    """Missing required args raises SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        main(["prepare-openie", "prepare"])
    assert exc.value.code == 2


def test_prepare_openie_valid_args() -> None:
    """Valid args parse correctly."""
    parser = _build_parser()
    args = parser.parse_args([
        "prepare-openie",
        "prepare",
        "--dataset", "musique",
        "--phase", "ner",
        "--workspace", "artifacts/openie/musique",
        "--config", "configs/experiment.yaml",
    ])
    assert args.openie_command == "prepare"
    assert args.dataset == "musique"
    assert args.phase == "ner"
    assert args.workspace == "artifacts/openie/musique"


# ── build-index ───────────────────────────────────────────────────────────────


def test_build_index_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["build-index", "--help"])
    assert exc.value.code == 0


def test_build_index_requires_dataset() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["build-index"])
    assert exc.value.code == 2


def test_build_index_parses_args() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "build-index",
        "--dataset", "musique",
        "--config", "configs/experiment.yaml",
    ])
    assert args.dataset == "musique"
    assert "experiment.yaml" in str(args.config)


# ── tune-gate ─────────────────────────────────────────────────────────────────


def test_tune_gate_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["tune-gate", "--help"])
    assert exc.value.code == 0


def test_tune_gate_requires_results_dir() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["tune-gate"])
    assert exc.value.code == 2


def test_tune_gate_parses_args() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "tune-gate",
        "--results-dir", "artifacts/runs",
        "--config", "configs/experiment.yaml",
    ])
    assert args.results_dir == "artifacts/runs"


# ── run ───────────────────────────────────────────────────────────────────────


def test_run_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0


def test_run_requires_method_and_dataset() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("method", _METHODS)
def test_run_method_choices(method: str) -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--method", method,
        "--dataset", "musique",
    ])
    assert args.method == method


@pytest.mark.parametrize("dataset", _DATASETS)
def test_run_dataset_choices(dataset: str) -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--method", "hipporag2",
        "--dataset", dataset,
    ])
    assert args.dataset == dataset


def test_run_parses_split_and_output() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "run",
        "--method", "metagate",
        "--dataset", "2wikimultihopqa",
        "--split", "dev",
        "--output-dir", "/tmp/run-out",
        "--gate-threshold", "configs/gate_threshold.json",
    ])
    assert args.method == "metagate"
    assert args.dataset == "2wikimultihopqa"
    assert args.split == "dev"
    assert args.output_dir == "/tmp/run-out"
    assert args.gate_threshold == "configs/gate_threshold.json"


# ── analyze ───────────────────────────────────────────────────────────────────


def test_analyze_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--help"])
    assert exc.value.code == 0


def test_analyze_requires_results_dir() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["analyze"])
    assert exc.value.code == 2


def test_analyze_parses_args() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "analyze",
        "--results-dir", "artifacts/runs",
        "--output-dir", "results",
        "--gate-threshold", "configs/gate_threshold.json",
    ])
    assert args.results_dir == "artifacts/runs"
    assert args.output_dir == "results"


# ── verify-run ────────────────────────────────────────────────────────────────


def test_verify_run_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["verify-run", "--help"])
    assert exc.value.code == 0


def test_verify_run_requires_run_dir() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["verify-run"])
    assert exc.value.code == 2


def test_verify_run_parses_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(["verify-run", "--run-dir", "artifacts/runs/hipporag2_nq_rear_test"])
    assert args.run_dir == "artifacts/runs/hipporag2_nq_rear_test"


# ── Integration: verify-run with real (temp) data ─────────────────────────────


def test_verify_run_passes_on_valid_run_dir(tmp_path: Path) -> None:
    """verify-run should pass when results.jsonl and run_manifest.json are valid."""
    from metagate_hipporag.models import Example, MethodResult

    run_dir = tmp_path / "hipporag2_nq_rear_test"
    run_dir.mkdir()

    # Write a valid result row
    example = Example(
        dataset="nq_rear",
        example_id="q1",
        question="What is X?",
        gold_answers=["answer"],
        gold_docs=["doc1"],
        stratum="default",
    )
    result = MethodResult(
        run_id="test-run-20260725-abc123",
        method="hipporag2",
        example=example,
        first_retrieval=None,
        second_retrieval=None,
        fused_passages=[],
        answer="answer",
        gate_decisions=[],
        expanded=False,
        abstain_flag=False,
        usage={},
        errors=[],
    )

    results_path = run_dir / "results.jsonl"
    results_path.write_text(
        result.model_dump_json() + "\n",
        encoding="utf-8",
    )

    # Write a matching manifest
    manifest = {
        "run_id": "test-run-20260725-abc123",
        "total_examples": 1,
        "effective_config_hash": "abc123",
        "method": "hipporag2",
        "dataset": "nq_rear",
        "split": "test",
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rc = main(["verify-run", "--run-dir", str(run_dir)])
    assert rc == 0


def test_verify_run_fails_on_missing_results(tmp_path: Path) -> None:
    """verify-run fails when results.jsonl is missing."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    rc = main(["verify-run", "--run-dir", str(run_dir)])
    assert rc == 1


def test_verify_run_fails_on_broken_jsonl(tmp_path: Path) -> None:
    """verify-run fails when results.jsonl has malformed rows."""
    run_dir = tmp_path / "broken_run"
    run_dir.mkdir()

    # Broken JSONL
    (run_dir / "results.jsonl").write_text("not json\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    rc = main(["verify-run", "--run-dir", str(run_dir)])
    assert rc == 1


# ── app entry point ───────────────────────────────────────────────────────────


def test_app_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Console-script entry point should call main()."""
    called: list[int] = []

    def fake_main() -> int:
        called.append(1)
        return 0

    # Monkeypatch both main (to track calls) and sys.exit (to avoid actual exit)
    monkeypatch.setattr("metagate_hipporag.cli.main", fake_main)

    # app() wraps main() in sys.exit(); catch that
    with pytest.raises(SystemExit) as exc:
        app()
    assert exc.value.code == 0
    assert called == [1]


# ── edge cases ────────────────────────────────────────────────────────────────


def test_unknown_subcommand() -> None:
    """Unknown subcommand exits with error."""
    with pytest.raises(SystemExit) as exc:
        main(["nonexistent"])
    assert exc.value.code == 2


def test_config_default_is_absolute() -> None:
    """Default config path should be absolute."""
    parser = _build_parser()
    args = parser.parse_args(["prepare-data"])
    config = Path(str(args.config))
    assert config.is_absolute()
    assert config.name == "experiment.yaml"


def test_prepare_openie_rejects_invalid_dataset() -> None:
    """Invalid dataset name should fail parsing."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "prepare-openie",
            "prepare",
            "--dataset", "invalid_dataset",
            "--phase", "ner",
            "--workspace", "ws",
        ])


def test_build_index_rejects_invalid_dataset() -> None:
    """Invalid dataset name should fail parsing."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "build-index",
            "--dataset", "invalid",
        ])
