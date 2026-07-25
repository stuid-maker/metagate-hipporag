"""Unified CLI for MetaGate-HippoRAG 2 experiments.

Subcommands
-----------
``prepare-data``     Download datasets and freeze train/dev/test splits.
``prepare-openie``   Run the two-stage Batch OpenIE pipeline (NER → Triple).
``build-index``      Build the fingerprinted HippoRAG graph index.
``tune-gate``        Tune the gate threshold on pooled dev-set results.
``run``              Execute one method on one dataset × split.
``analyze``          Compute evaluation metrics and statistical tests.
``verify-run``       Check a completed run manifest for integrity.

Entry point (also registered as console script ``metagate``)::

    python -m metagate_hipporag.cli <subcommand> [options]
    metagate <subcommand> [options]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# ── shared helpers ────────────────────────────────────────────────────────────

_DATASETS = ["nq_rear", "musique", "2wikimultihopqa"]
_METHODS = ["llm_only", "dense_rag", "hipporag2", "always_expand", "metagate"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config() -> Path:
    return _repo_root() / "configs" / "experiment.yaml"


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=str(_default_config()),
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )


# ── prepare-data ──────────────────────────────────────────────────────────────


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    """Download datasets and freeze the dev/test splits."""
    from .data import prepare

    prepare(str(args.config))
    return 0


# ── prepare-openie ────────────────────────────────────────────────────────────


def _cmd_prepare_openie(args: argparse.Namespace) -> int:
    """Delegate to the batch_openie module CLI."""
    from .batch_openie import main as openie_main

    # Reconstruct argv for the batch_openie argparser
    forwarded = [
        args.openie_command,
        "--dataset", args.dataset,
        "--phase", args.phase,
        "--workspace", args.workspace,
    ]
    if args.config is not None:
        forwarded.extend(["--config", str(args.config)])
    if args.source is not None:
        forwarded.extend(["--source", str(args.source)])

    return openie_main(forwarded)


# ── build-index ───────────────────────────────────────────────────────────────


def _cmd_build_index(args: argparse.Namespace) -> int:
    """Build a fingerprinted HippoRAG index for one dataset."""
    from .config import load_config
    from .embedding import PersistentOpenAIEmbeddingModel, inject_embedding_model
    from .provenance import UsageLedger, index_config_hash, index_directory, sha256_file

    config = load_config(args.config)
    repo = _repo_root()

    # --- validate prerequisites -----------------------------------------------
    dataset = args.dataset

    # Corpus path
    corpus_path = repo / "data" / "raw" / f"{dataset}_corpus.json"
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}.  Run prepare-data first.", file=sys.stderr)
        return 1

    corpus_sha = sha256_file(corpus_path)

    # OpenIE results
    openie_path = (
        repo / "artifacts" / "openie" / dataset
        / "openie_results_ner_gpt-4o-mini-2024-07-18.json"
    )
    if not openie_path.exists():
        print(
            f"OpenIE results not found: {openie_path}.  Run prepare-openie first.",
            file=sys.stderr,
        )
        return 1

    openie_sha = sha256_file(openie_path)

    # --- compute hashes and directory -----------------------------------------
    upstream_sha = config.project.upstream_commit
    llm_slug = config.models.llm
    embedding_slug = config.models.embedding

    # OpenIE prompt hash — use corpus + OpenIE result as proxy since
    # the prompt content hash is tracked inside batch_openie's sidecar.
    # In a full run the gate/qa prompts are also frozen; we reuse config_hash here.
    openie_prompt_sha = openie_sha  # stable proxy

    index_cfg_hash = index_config_hash(
        upstream_sha=upstream_sha,
        upstream_version=config.project.upstream_package_version,
        preprocessing_version=config.project.preprocessing_version,
        llm_model=llm_slug,
        embedding_model=embedding_slug,
        embedding_dimensions=config.models.embedding_dimensions,
        instruction_mode=config.models.embedding_instruction_mode,
        linking_top_k=config.retrieval.linking_top_k,
        ppr_damping=config.retrieval.ppr_damping,
        passage_node_weight=config.retrieval.passage_node_weight,
        synonym_threshold=config.retrieval.synonym_threshold,
        openie_prompt_sha=openie_prompt_sha,
        corpus_sha=corpus_sha,
    )

    index_dir = index_directory(
        dataset=dataset,
        corpus_sha256=corpus_sha,
        upstream_sha=upstream_sha,
        llm_slug=llm_slug,
        embedding_slug=embedding_slug,
        openie_prompt_sha256=openie_prompt_sha,
        index_config_sha256=index_cfg_hash,
    )

    # Check for an existing completed manifest
    manifest_path = index_dir / "index_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            print(f"Index already built: {index_dir}", file=sys.stderr)
            print(f"  config_hash = {index_cfg_hash}", file=sys.stderr)
            return 0
        else:
            print(f"WARNING: stale partial index at {index_dir}", file=sys.stderr)
            print("Remove it manually and re-run if you want a fresh build.", file=sys.stderr)
            return 1

    # --- initialise ledger ----------------------------------------------------
    ledger_path = repo / "artifacts" / "ledger" / "build_index.jsonl"
    ledger = UsageLedger(ledger_path)

    # --- build embedding model ------------------------------------------------
    print(f"Building index for {dataset} → {index_dir}", file=sys.stderr)

    embedding_cache_path = repo / "artifacts" / "cache" / "embeddings.db"
    emb_global_config = SimpleNamespace(
        embedding_model_name=embedding_slug,
        embedding_return_as_normalized=True,
        embedding_max_seq_len=8192,
        embedding_batch_size=64,
        embedding_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        azure_embedding_endpoint=None,
    )
    embedding_model = PersistentOpenAIEmbeddingModel(
        emb_global_config,
        cache_db_path=embedding_cache_path,
        dimensions=config.models.embedding_dimensions,
        instruction_mode=config.models.embedding_instruction_mode,
        ledger=ledger,
        price_per_million=config.pricing_snapshot.text_embedding_3_large_per_million,
    )

    # --- lazy-import hipporag (requires upstream bootstrap) --------------------
    try:
        from hipporag import HippoRAG  # type: ignore[import-untyped]
    except ImportError:
        print(
            "HippoRAG not importable.  Run scripts/bootstrap_upstream.py first.",
            file=sys.stderr,
        )
        return 1

    # --- instantiate engine ---------------------------------------------------
    # The upstream HippoRAG constructor accepts a global_config dict.
    # We need to build one from our config.
    global_config = {
        "llm_name": llm_slug,
        "llm_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "embedding_model_name": embedding_slug,
        "embedding_dim": config.models.embedding_dimensions,
        "embedding_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "linking_top_k": config.retrieval.linking_top_k,
        "ppr_damping": config.retrieval.ppr_damping,
        "passage_node_weight": config.retrieval.passage_node_weight,
        "synonymity_threshold": config.retrieval.synonym_threshold,
        "save_dir": str(index_dir),
        "corpus_path": str(corpus_path),
        "openie_path": str(openie_path),
        "dataset_name": dataset,
    }

    engine = HippoRAG(global_config=global_config)

    # Inject our persistent embedding model
    inject_embedding_model(engine, embedding_model)

    # --- build index ----------------------------------------------------------
    print("Indexing ...", file=sys.stderr)
    engine.index()

    # --- write manifest -------------------------------------------------------
    import time

    manifest = {
        "dataset": dataset,
        "corpus_sha256": corpus_sha,
        "openie_sha256": openie_sha,
        "upstream_sha": upstream_sha,
        "upstream_version": config.project.upstream_package_version,
        "index_config_hash": index_cfg_hash,
        "llm_model": llm_slug,
        "embedding_model": embedding_slug,
        "embedding_dimensions": config.models.embedding_dimensions,
        "instruction_mode": config.models.embedding_instruction_mode,
        "preprocessing_version": config.project.preprocessing_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": True,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Index built successfully: {index_dir}", file=sys.stderr)

    return 0


# ── tune-gate ─────────────────────────────────────────────────────────────────


def _cmd_tune_gate(args: argparse.Namespace) -> int:
    """Tune the gate threshold on pooled dev-set results."""
    from .config import load_config
    from .metagate import select_threshold

    config = load_config(args.config)
    results_dir = Path(args.results_dir)

    all_probs: list[float] = []
    all_sufficient: list[bool] = []

    for dataset in _DATASETS:
        results_path = results_dir / f"metagate_{dataset}_dev" / "results.jsonl"
        if not results_path.exists():
            print(
                f"WARNING: missing dev results for {dataset} at {results_path}",
                file=sys.stderr,
            )
            continue

        from .provenance import read_jsonl_recover_tail

        rows = read_jsonl_recover_tail(results_path)
        for row in rows:
            from .models import MethodResult
            try:
                result = MethodResult.model_validate(row)
            except Exception:
                continue

            if result.gate_decisions:
                g = result.gate_decisions[0]
                all_probs.append(g.evidence_sufficient_probability)

                # Ground truth: Recall@5 == 1 on first retrieval
                if result.first_retrieval is not None:
                    from .evaluation import compute_recall

                    r5 = compute_recall(
                        result.first_retrieval.passages,
                        result.example.gold_docs,
                    ).get("Recall@5", 0.0)
                    all_sufficient.append(r5 >= 0.999)
                else:
                    all_sufficient.append(False)

    if not all_probs:
        print("No dev results found to tune gate.  Run the dev experiments first.", file=sys.stderr)
        return 1

    print(
        f"Tuning gate on {len(all_probs)} dev examples across {_DATASETS}",
        file=sys.stderr,
    )

    threshold = select_threshold(
        all_probs,
        all_sufficient,
        list(config.gate.threshold_candidates),
    )

    # Persist threshold
    threshold_path = Path(args.output) if args.output else _repo_root() / "configs" / "gate_threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_data = {
        "threshold": threshold,
        "config_hash": config.config_hash,
        "n_examples": len(all_probs),
        "candidates": list(config.gate.threshold_candidates),
    }
    threshold_path.write_text(
        json.dumps(threshold_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected threshold: {threshold}", file=sys.stderr)
    print(f"Saved to: {threshold_path}", file=sys.stderr)

    return 0


# ── run ───────────────────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute one method on one dataset × split."""
    from .config import load_config
    from .models import DatasetId, MethodId

    config = load_config(args.config)
    repo = _repo_root()

    method: MethodId = args.method  # type: ignore[assignment]
    dataset: DatasetId = args.dataset  # type: ignore[assignment]
    split = args.split

    # --- load examples -------------------------------------------------------
    splits_path = repo / "data" / "splits" / f"{dataset}.json"
    if not splits_path.exists():
        print(f"Split file not found: {splits_path}. Run prepare-data first.", file=sys.stderr)
        return 1

    split_data = json.loads(splits_path.read_text(encoding="utf-8"))
    ids = split_data["dev_ids"] if split == "dev" else split_data["test_ids"]

    # Load raw examples
    raw_path = repo / "data" / "raw" / f"{dataset}.json"
    raw_data = json.loads(raw_path.read_text(encoding="utf-8"))

    from .data import normalize_example
    from .models import Example

    examples: list[Example] = []
    for i, raw in enumerate(raw_data):
        fallback_id = f"{dataset}_{i}" if dataset == "nq_rear" else None
        ex = normalize_example(dataset, raw, fallback_id=fallback_id)
        if ex.example_id in ids:
            examples.append(ex)

    # Maintain order by ids
    id_order = {eid: idx for idx, eid in enumerate(ids)}
    examples.sort(key=lambda ex: id_order.get(ex.example_id, 9999))

    print(f"Loaded {len(examples)} {split} examples for {dataset}", file=sys.stderr)

    # --- initialise components ------------------------------------------------
    ledger_path = repo / "artifacts" / "ledger" / f"run_{method}_{dataset}_{split}.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    from .provenance import UsageLedger
    ledger = UsageLedger(ledger_path)

    cache_path = repo / "artifacts" / "cache" / "completions.db"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    from .openai_client import CachedStructuredClient
    client = CachedStructuredClient(
        cache_path=cache_path,
        input_price_per_million=config.pricing_snapshot.gpt_4o_mini_input_per_million,
        output_price_per_million=config.pricing_snapshot.gpt_4o_mini_output_per_million,
        ledger=ledger,
        project_limit_usd=config.budget.project_max_actual_usd,
    )

    # --- bridge / metagate (for retrieval-based methods) ---------------------
    bridge = None
    metagate = None
    gate_threshold = None

    if method in ("dense_rag", "hipporag2", "always_expand", "metagate"):
        # Load gate threshold if needed
        if method in ("always_expand", "metagate"):
            threshold_path = (
                Path(args.gate_threshold)
                if args.gate_threshold
                else repo / "configs" / "gate_threshold.json"
            )
            if not threshold_path.exists():
                print(
                    "Gate threshold not found. Run tune-gate first, or provide --gate-threshold.",
                    file=sys.stderr,
                )
                return 1
            th_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            gate_threshold = float(th_data["threshold"])
            print(f"Using gate threshold: {gate_threshold}", file=sys.stderr)

        # Build index directory
        from .provenance import index_config_hash, index_directory, sha256_file

        corpus_path = repo / "data" / "raw" / f"{dataset}_corpus.json"
        corpus_sha = sha256_file(corpus_path)

        openie_path = (
            repo / "artifacts" / "openie" / dataset
            / "openie_results_ner_gpt-4o-mini-2024-07-18.json"
        )
        openie_sha = sha256_file(openie_path) if openie_path.exists() else "0" * 64

        index_cfg_hash = index_config_hash(
            upstream_sha=config.project.upstream_commit,
            upstream_version=config.project.upstream_package_version,
            preprocessing_version=config.project.preprocessing_version,
            llm_model=config.models.llm,
            embedding_model=config.models.embedding,
            embedding_dimensions=config.models.embedding_dimensions,
            instruction_mode=config.models.embedding_instruction_mode,
            linking_top_k=config.retrieval.linking_top_k,
            ppr_damping=config.retrieval.ppr_damping,
            passage_node_weight=config.retrieval.passage_node_weight,
            synonym_threshold=config.retrieval.synonym_threshold,
            openie_prompt_sha=openie_sha,
            corpus_sha=corpus_sha,
        )

        index_dir = index_directory(
            dataset=dataset,
            corpus_sha256=corpus_sha,
            upstream_sha=config.project.upstream_commit,
            llm_slug=config.models.llm,
            embedding_slug=config.models.embedding,
            openie_prompt_sha256=openie_sha,
            index_config_sha256=index_cfg_hash,
        )

        # Check index exists
        if not index_dir.exists():
            print(f"Index not found: {index_dir}. Run build-index first.", file=sys.stderr)
            return 1

        # Initialise engine
        try:
            from hipporag import HippoRAG  # type: ignore[import-untyped]
        except ImportError:
            print("HippoRAG not importable. Run bootstrap first.", file=sys.stderr)
            return 1

        global_config = {
            "llm_name": config.models.llm,
            "llm_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "embedding_model_name": config.models.embedding,
            "embedding_dim": config.models.embedding_dimensions,
            "embedding_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "linking_top_k": config.retrieval.linking_top_k,
            "ppr_damping": config.retrieval.ppr_damping,
            "passage_node_weight": config.retrieval.passage_node_weight,
            "synonymity_threshold": config.retrieval.synonym_threshold,
            "save_dir": str(index_dir),
            "corpus_path": str(corpus_path),
            "openie_path": str(openie_path),
            "dataset_name": dataset,
        }

        engine = HippoRAG(global_config=global_config)

        from .embedding import PersistentOpenAIEmbeddingModel, inject_embedding_model
        embedding_cache = repo / "artifacts" / "cache" / "embeddings.db"
        emb_global_config_run = SimpleNamespace(
            embedding_model_name=config.models.embedding,
            embedding_return_as_normalized=True,
            embedding_max_seq_len=8192,
            embedding_batch_size=64,
            embedding_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            azure_embedding_endpoint=None,
        )
        emb_model = PersistentOpenAIEmbeddingModel(
            emb_global_config_run,
            cache_db_path=embedding_cache,
            dimensions=config.models.embedding_dimensions,
            instruction_mode=config.models.embedding_instruction_mode,
            ledger=ledger,
            price_per_million=config.pricing_snapshot.text_embedding_3_large_per_million,
        )
        inject_embedding_model(engine, emb_model)

        from .hipporag_adapter import HippoRAGBridge
        bridge = HippoRAGBridge(engine, top_k=config.retrieval.output_top_k)

        if method in ("always_expand", "metagate"):
            from .metagate import MetaGate
            metagate = MetaGate(
                client=client,
                threshold=gate_threshold or 0.75,
            )

    # --- compute effective config hash ---------------------------------------
    from .provenance import sha256_file as _sha256
    gate_prompt_path = repo / "configs" / "gate_prompt.json"
    qa_prompt_path = repo / "configs" / "qa_prompt.json"
    patch_path = repo / "patches" / "hipporag-openai-only.patch"

    frozen_hashes = {
        "compatibility_patch": _sha256(patch_path) if patch_path.exists() else "0" * 64,
        "data_manifest": _sha256(splits_path),
        "gate_prompt": _sha256(gate_prompt_path),
        "qa_prompt": _sha256(qa_prompt_path),
        "openie_ner_prompt": "0" * 64,  # placeholder — real hash in batch_openie sidecar
        "openie_triple_prompt": "0" * 64,
    }

    from .config import effective_config_hash
    eff_hash = effective_config_hash(config, frozen_hashes)

    # --- output dir ----------------------------------------------------------
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo / "artifacts" / "runs" / f"{method}_{dataset}_{split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- run ------------------------------------------------------------------
    from .methods import run_method

    print(f"Running {method} on {dataset}/{split} ({len(examples)} examples)", file=sys.stderr)
    results = run_method(
        method=method,
        examples=examples,
        bridge=bridge,
        metagate=metagate,
        client=client,
        ledger=ledger,
        output_dir=output_dir,
        effective_config_hash=eff_hash,
        gate_threshold=gate_threshold,
        rrf_k=config.retrieval.rrf_k,
        top_k=config.retrieval.output_top_k,
        llm_model=config.models.llm,
        seed=config.project.seed,
        temperature=config.models.temperature,
        max_tokens=config.models.qa_max_output_tokens,
    )

    print(f"Completed {len(results)} results → {output_dir}", file=sys.stderr)
    return 0


# ── analyze ───────────────────────────────────────────────────────────────────


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Compute evaluation metrics and statistical tests on run results."""
    from .config import load_config
    from .evaluation import evaluate
    from .models import DatasetId, MethodId, MethodResult
    from .provenance import read_jsonl_recover_tail

    config = load_config(args.config)
    results_dir = Path(args.results_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _repo_root() / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load all results ----------------------------------------------------
    all_results: dict[DatasetId, dict[MethodId, list[MethodResult]]] = {}
    for dataset in _DATASETS:
        all_results[dataset] = {}
        for method in _METHODS:
            path = results_dir / f"{method}_{dataset}_test" / "results.jsonl"
            if not path.exists():
                print(
                    f"WARNING: missing results for {method}/{dataset} at {path}",
                    file=sys.stderr,
                )
                continue
            rows = read_jsonl_recover_tail(path)
            parsed: list[MethodResult] = []
            for row in rows:
                try:
                    parsed.append(MethodResult.model_validate(row))
                except Exception:
                    continue
            if parsed:
                all_results[dataset][method] = parsed

    # --- gate threshold ------------------------------------------------------
    threshold_path = (
        Path(args.gate_threshold)
        if args.gate_threshold
        else _repo_root() / "configs" / "gate_threshold.json"
    )
    if threshold_path.exists():
        th_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        gate_threshold = float(th_data["threshold"])
    else:
        gate_threshold = 0.75
        print(f"WARNING: no gate_threshold.json, using default {gate_threshold}", file=sys.stderr)

    # --- evaluate each method × dataset --------------------------------------
    eval_rows: list[dict[str, Any]] = []
    for dataset in _DATASETS:
        methods_in_dataset = all_results.get(dataset, {})
        always_expand_res = methods_in_dataset.get("always_expand")  # type: ignore[arg-type]

        for method, results in methods_in_dataset.items():
            report = evaluate(
                results,
                gate_threshold=gate_threshold,
                ece_bins=config.statistics.ece_bins,
                always_expand_results=always_expand_res if method != "always_expand" else None,
            )
            eval_rows.append(report.to_dict())

    # Write evaluation table
    import pandas as pd
    eval_df = pd.DataFrame(eval_rows)
    eval_csv = output_dir / "tables" / "evaluation.csv"
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(eval_csv, index=False)
    print(f"Evaluation table → {eval_csv}", file=sys.stderr)

    # --- statistics ----------------------------------------------------------
    from .statistics import run_statistics

    comparisons: list[tuple[MethodId, MethodId]] = [
        ("hipporag2", "dense_rag"),
        ("metagate", "hipporag2"),
        ("metagate", "always_expand"),
    ]
    metrics = ["recall_at_5", "em", "token_f1", "method_equivalent_cost_usd", "llm_calls"]

    results_for_stats: dict[DatasetId, dict[MethodId, list[MethodResult]]] = {}
    for dataset in _DATASETS:
        if dataset in all_results:
            results_for_stats[dataset] = all_results[dataset]  # type: ignore[assignment]

    stats_df = run_statistics(
        results_for_stats,
        comparisons=comparisons,
        metrics=metrics,
        bootstrap_seed=config.statistics.bootstrap_seed,
        n_resamples=config.statistics.bootstrap_resamples,
        confidence_level=config.statistics.confidence_level,
        alpha=0.05,
        noninferiority_margin=config.statistics.noninferiority_margin_token_f1,
    )

    stats_csv = output_dir / "tables" / "statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Statistics table → {stats_csv}", file=sys.stderr)

    return 0


# ── verify-run ────────────────────────────────────────────────────────────────


def _cmd_verify_run(args: argparse.Namespace) -> int:
    """Verify a completed run manifest for integrity."""
    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "run_manifest.json"

    issues: list[str] = []

    # Check results.jsonl exists
    if not results_path.exists():
        issues.append(f"Missing results.jsonl at {results_path}")

    # Check manifest exists
    if not manifest_path.exists():
        issues.append(f"Missing run_manifest.json at {manifest_path}")

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}", file=sys.stderr)
        return 1

    # Validate JSONL integrity
    from .provenance import read_jsonl_recover_tail

    rows = read_jsonl_recover_tail(results_path)
    if not rows:
        issues.append("results.jsonl is empty or unparseable")

    # Parse and validate each row
    from .models import MethodResult

    parsed: list[MethodResult] = []
    for i, row in enumerate(rows):
        try:
            parsed.append(MethodResult.model_validate(row))
        except Exception as exc:
            issues.append(f"Row {i}: validation failed: {exc}")

    # Check manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_count = manifest.get("total_examples")
        if expected_count is not None and len(parsed) != expected_count:
            issues.append(
                f"Row count mismatch: manifest={expected_count}, results={len(parsed)}"
            )
        expected_hash = manifest.get("effective_config_hash")
        if expected_hash and parsed:
            actual_hash = parsed[0].run_id
            # run_id is derived from effective_config_hash in the runner
            if expected_hash not in actual_hash:
                issues.append(
                    f"Config hash mismatch: manifest has {expected_hash[:12]}..., "
                    f"results have {actual_hash[:12]}..."
                )
    except Exception as exc:
        issues.append(f"Manifest read error: {exc}")

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}", file=sys.stderr)
        return 1

    print(
        f"PASS: {len(parsed)} valid results, manifest consistent  ({run_dir})",
        file=sys.stderr,
    )
    return 0


# ── main entry point ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metagate",
        description="MetaGate-HippoRAG 2 experiment CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # -- prepare-data ----------------------------------------------------------
    p_data = sub.add_parser("prepare-data", help="Download data and freeze splits")
    _add_config(p_data)
    p_data.set_defaults(func=_cmd_prepare_data)

    # -- prepare-openie --------------------------------------------------------
    p_openie = sub.add_parser(
        "prepare-openie",
        help="Run the two-stage Batch OpenIE pipeline (NER → Triple)",
    )
    p_openie.add_argument(
        "openie_command",
        choices=["prepare", "submit", "poll", "collect"],
        help="OpenIE pipeline stage",
    )
    p_openie.add_argument("--dataset", required=True, choices=_DATASETS)
    p_openie.add_argument("--phase", required=True, choices=["ner", "triple"])
    p_openie.add_argument("--workspace", required=True, help="e.g. artifacts/openie/musique")
    p_openie.add_argument("--source", help="Path to JSON mapping chunk_id → passage text")
    _add_config(p_openie)
    p_openie.set_defaults(func=_cmd_prepare_openie)

    # -- build-index -----------------------------------------------------------
    p_index = sub.add_parser("build-index", help="Build HippoRAG graph index")
    p_index.add_argument("--dataset", required=True, choices=_DATASETS)
    _add_config(p_index)
    p_index.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env)")
    p_index.set_defaults(func=_cmd_build_index)

    # -- tune-gate -------------------------------------------------------------
    p_gate = sub.add_parser("tune-gate", help="Tune gate threshold on dev results")
    p_gate.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing dev results (e.g. artifacts/runs/)",
    )
    _add_config(p_gate)
    p_gate.add_argument("--output", help="Path to save gate_threshold.json")
    p_gate.set_defaults(func=_cmd_tune_gate)

    # -- run -------------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run one method on a dataset")
    p_run.add_argument("--method", required=True, choices=_METHODS)
    p_run.add_argument("--dataset", required=True, choices=_DATASETS)
    p_run.add_argument("--split", default="test", choices=["dev", "test"])
    _add_config(p_run)
    p_run.add_argument(
        "--output-dir",
        help="Output directory (default: artifacts/runs/<method>_<dataset>_<split>/)",
    )
    p_run.add_argument("--gate-threshold", help="Path to gate_threshold.json")
    p_run.add_argument("--api-key", help="OpenAI API key")
    p_run.set_defaults(func=_cmd_run)

    # -- analyze ---------------------------------------------------------------
    p_analyze = sub.add_parser("analyze", help="Compute metrics and statistics")
    p_analyze.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing run results (e.g. artifacts/runs/)",
    )
    _add_config(p_analyze)
    p_analyze.add_argument("--output-dir", help="Output directory for tables (default: results/)")
    p_analyze.add_argument("--gate-threshold", help="Path to gate_threshold.json")
    p_analyze.set_defaults(func=_cmd_analyze)

    # -- verify-run ------------------------------------------------------------
    p_verify = sub.add_parser("verify-run", help="Check run manifest integrity")
    p_verify.add_argument("--run-dir", required=True, help="Path to a run output directory")
    p_verify.set_defaults(func=_cmd_verify_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m metagate_hipporag.cli``."""
    parser = _build_parser()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


# Console-script entry point (metagate = metagate_hipporag.cli:app)
def app() -> None:
    """Console-script entry point — no arguments are passed."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
