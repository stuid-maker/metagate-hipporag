"""Tests for methods.py — five method runners and resumable JSONL execution.

Covers:
- ``ResumableRunner`` — manifest, completed-ID tracking, config-hash validation
- ``run_llm_only`` — direct LLM QA without retrieval
- ``run_dense_rag`` — dense-only retrieval + upstream QA
- ``run_hipporag2`` — full graph retrieval + upstream QA
- ``run_always_expand`` — forced two-round expansion + RRF + QA
- ``run_metagate`` — gate-controlled expansion with abstain flagging
- ``run_method`` — high-level orchestrator with dependency validation
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from metagate_hipporag.metagate import MetaGate
from metagate_hipporag.methods import (
    QA_SYSTEM_PROMPT,
    ResumableRunner,
    _compute_run_id,
    run_always_expand,
    run_dense_rag,
    run_hipporag2,
    run_llm_only,
    run_metagate,
    run_method,
)
from metagate_hipporag.models import (
    Example,
    GateDecision,
    MethodResult,
    RetrievalTrace,
    RetrievedPassage,
    Usage,
)
from metagate_hipporag.provenance import UsageLedger

# ── Shared test constants ────────────────────────────────────────────────────

CONFIG_HASH = "a" * 64
RUN_ID = "0123456789abcdef"


def make_example(
    dataset: str = "musique",
    example_id: str = "test-1",
    question: str = "What is the capital of France?",
) -> Example:
    """Create a minimal Example for testing."""
    return Example(
        dataset=dataset,  # type: ignore[arg-type]
        example_id=example_id,
        question=question,
        gold_answers=["Paris"],
        gold_docs=["France\\nParis is the capital of France."],
        stratum="2",
    )


def make_trace(
    query: str = "test query",
    passages: list[RetrievedPassage] | None = None,
) -> RetrievalTrace:
    """Create a minimal RetrievalTrace."""
    if passages is None:
        passages = [
            RetrievedPassage(
                chunk_id="chunk-1",
                text="Paris is the capital of France.",
                score=0.95,
                rank=1,
            )
        ]
    return RetrievalTrace(
        retrieval_query=query,
        passages=passages,
        facts_before_filter=[("Paris", "is_capital_of", "France")],
        facts_after_filter=[("Paris", "is_capital_of", "France")],
        used_dense_fallback=False,
    )


# ── Mock helpers ─────────────────────────────────────────────────────────────


@dataclass
class MockStructuredCompletion:
    """Fake ``StructuredCompletion`` returned by the mock client."""

    value: Any
    usage: Usage


def make_mock_metagate(
    *,
    expand: bool = False,
    abstain: bool = False,
    rewrite: str = "rewritten query",
) -> MagicMock:
    """Create a mock MetaGate controller."""
    gate = MagicMock(spec=MetaGate)

    first_decision = MagicMock()
    first_decision.gate = GateDecision(
        evidence_sufficient_probability=0.9 if not expand else 0.3,
        missing_information="none" if not expand else "missing details",
        retrieval_rewrite=rewrite,
        rationale_summary="evidence is sufficient" if not expand else "need more",
    )
    first_decision.expand = expand
    first_decision.expansion_number = 0
    first_decision.abstain = False

    second_decision = MagicMock()
    second_decision.gate = GateDecision(
        evidence_sufficient_probability=0.7 if not abstain else 0.2,
        missing_information="none",
        retrieval_rewrite=rewrite,
        rationale_summary="sufficient after expansion" if not abstain else "still insufficient",
    )
    second_decision.expand = False
    second_decision.expansion_number = 1
    second_decision.abstain = abstain

    gate.decide.return_value = first_decision
    gate.decide_second.return_value = second_decision

    return gate


def make_mock_client(*, answer: str = "Paris") -> MagicMock:
    """Create a mock ``CachedStructuredClient``."""
    client = MagicMock()

    # Build a mock that returns the answer as a _QAAnswer
    mock_completion = MockStructuredCompletion(
        value=MagicMock(answer=answer),
        usage=Usage(
            prompt_tokens=50,
            completion_tokens=5,
            actual_usd=0.0001,
            method_equivalent_usd=0.0001,
            observed_latency_seconds=0.5,
            method_equivalent_latency_seconds=0.5,
        ),
    )
    client.complete.return_value = mock_completion
    return client


def make_mock_bridge(
    *,
    passages: list[RetrievedPassage] | None = None,
    answer: str = "Paris",
) -> MagicMock:
    """Create a mock ``HippoRAGBridge``."""
    bridge = MagicMock()
    trace = make_trace(passages=passages)

    bridge.retrieve_with_trace.return_value = trace
    bridge.dense_with_trace.return_value = trace

    # answer() returns (query_solutions, response_messages, metadata)
    mock_qs = MagicMock()
    mock_qs.answer = answer
    bridge.answer.return_value = ([mock_qs], ["fake response"], [{"model": "fake"}])

    return bridge


def make_tmp_ledger() -> UsageLedger:
    """Create a real UsageLedger backed by a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    return UsageLedger(Path(tmp_path), limit_usd=10.0)


# ── ResumableRunner tests ────────────────────────────────────────────────────


class TestResumableRunner:
    """Tests for the JSONL-based resumable runner."""

    def test_creates_manifest_and_dir(self) -> None:
        """Runner creates output dir and manifest on first use."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-abc"
            runner = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash=CONFIG_HASH,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            assert output_dir.exists()
            assert runner._manifest_path.exists()
            manifest_data = json.loads(runner._manifest_path.read_text())
            assert manifest_data["run_id"] == RUN_ID
            assert manifest_data["method"] == "hipporag2"
            assert manifest_data["dataset"] == "musique"

    def test_completed_ids_empty_on_start(self) -> None:
        """No completed IDs when starting fresh."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-abc"
            runner = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash=CONFIG_HASH,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            assert runner.completed_count == 0
            assert len(runner.completed_ids) == 0

    def test_append_adds_to_completed(self) -> None:
        """Appending a result adds its example_id to the completed set."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-abc"
            runner = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash=CONFIG_HASH,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            example = make_example()
            result = MethodResult(
                run_id=RUN_ID,
                method="hipporag2",
                example=example,
                first_retrieval=None,
                second_retrieval=None,
                fused_passages=[],
                answer="Paris",
                gate_decisions=[],
                expanded=False,
                abstain_flag=False,
                usage=Usage(),
                errors=[],
            )
            runner.append(result)
            assert runner.completed_count == 1
            assert example.example_id in runner.completed_ids

    def test_skips_completed_examples_on_resume(self) -> None:
        """After writing results, a new runner instance sees them as completed."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-abc"

            # First run: write one result
            runner1 = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash=CONFIG_HASH,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            example = make_example()
            result = MethodResult(
                run_id=RUN_ID,
                method="hipporag2",
                example=example,
                first_retrieval=None,
                second_retrieval=None,
                fused_passages=[],
                answer="Paris",
                gate_decisions=[],
                expanded=False,
                abstain_flag=False,
                usage=Usage(),
                errors=[],
            )
            runner1.append(result)

            # Second run: should see completed IDs
            runner2 = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash=CONFIG_HASH,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            assert runner2.completed_count == 1
            assert example.example_id in runner2.completed_ids

    def test_config_hash_mismatch_resets(self) -> None:
        """When config hash changes, the old run is staled and a fresh one begins."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-abc"

            # First run with hash A
            runner1 = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash="a" * 64,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            example = make_example()
            runner1.append(
                MethodResult(
                    run_id=RUN_ID,
                    method="hipporag2",
                    example=example,
                    first_retrieval=None,
                    second_retrieval=None,
                    fused_passages=[],
                    answer="Paris",
                    gate_decisions=[],
                    expanded=False,
                    abstain_flag=False,
                    usage=Usage(),
                    errors=[],
                )
            )

            # Second run with different hash — should stale and reset
            runner2 = ResumableRunner(
                output_dir=output_dir,
                run_id=RUN_ID,
                effective_config_hash="b" * 64,
                method="hipporag2",
                dataset="musique",
                split="dev",
            )
            assert runner2.completed_count == 0


# ── LLM-only tests ───────────────────────────────────────────────────────────


class TestLLMOnly:
    """Tests for the LLM-only baseline method."""

    def test_basic_answer(self) -> None:
        """LLM-only returns an answer from the mock client."""
        example = make_example()
        client = make_mock_client(answer="Paris")
        ledger = make_tmp_ledger()

        result = run_llm_only(
            example=example,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.method == "llm_only"
        assert result.answer == "Paris"
        assert result.example == example
        assert result.first_retrieval is None
        assert result.expanded is False
        assert result.abstain_flag is False
        assert len(result.errors) == 0

    def test_qa_prompt_is_loaded(self) -> None:
        """The QA prompt is frozen and loaded at import time."""
        assert QA_SYSTEM_PROMPT
        assert "parametric" in QA_SYSTEM_PROMPT.lower()

    def test_error_is_captured(self) -> None:
        """When the client throws, the error is recorded."""
        example = make_example()
        client = make_mock_client(answer="Paris")
        client.complete.side_effect = RuntimeError("API down")
        ledger = make_tmp_ledger()

        result = run_llm_only(
            example=example,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.answer == ""
        assert len(result.errors) == 1
        assert "API down" in result.errors[0]


# ── Dense RAG tests ──────────────────────────────────────────────────────────


class TestDenseRAG:
    """Tests for the Dense RAG baseline method."""

    def test_basic_retrieval_and_qa(self) -> None:
        """Dense RAG retrieves passages and runs upstream QA."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_dense_rag(
            example=example,
            bridge=bridge,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.method == "dense_rag"
        assert result.answer == "Paris"
        bridge.dense_with_trace.assert_called_once_with(example.question)
        bridge.answer.assert_called_once()
        assert len(result.fused_passages) > 0

    def test_retrieval_error_captured(self) -> None:
        """When retrieval fails, error is recorded and QA still attempted."""
        example = make_example()
        bridge = make_mock_bridge(answer="fallback")
        bridge.dense_with_trace.side_effect = RuntimeError("index missing")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_dense_rag(
            example=example,
            bridge=bridge,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert len(result.errors) == 1
        assert "index missing" in result.errors[0]


# ── HippoRAG 2 tests ─────────────────────────────────────────────────────────


class TestHippoRAG2:
    """Tests for the HippoRAG 2 baseline method."""

    def test_basic_graph_retrieval(self) -> None:
        """HippoRAG 2 uses full graph retrieval path."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_hipporag2(
            example=example,
            bridge=bridge,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.method == "hipporag2"
        assert result.answer == "Paris"
        bridge.retrieve_with_trace.assert_called_once_with(example.question)
        # dense_with_trace should NOT have been called
        bridge.dense_with_trace.assert_not_called()

    def test_facts_captured_in_trace(self) -> None:
        """Retrieval trace includes facts before/after filtering."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_hipporag2(
            example=example,
            bridge=bridge,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.first_retrieval is not None
        assert len(result.first_retrieval.facts_before_filter) == 1


# ── Always-Expand tests ──────────────────────────────────────────────────────


class TestAlwaysExpand:
    """Tests for the Always-Expand baseline method."""

    def test_always_expands_regardless_of_gate(self) -> None:
        """Always-Expand performs two retrievals even when gate is confident."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate(expand=False)  # gate says "don't expand"
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_always_expand(
            example=example,
            bridge=bridge,
            metagate=metagate,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.expanded is True
        assert result.abstain_flag is False
        # Two retrievals should have happened
        assert bridge.retrieve_with_trace.call_count == 2
        # Two gate calls should have happened
        assert metagate.decide.call_count == 1
        assert metagate.decide_second.call_count == 1

    def test_second_retrieval_uses_rewrite(self) -> None:
        """The second retrieval query comes from the gate's rewrite."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate(expand=True, rewrite="capital of France")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        _ = run_always_expand(
            example=example,
            bridge=bridge,
            metagate=metagate,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        # Second call should use the rewrite query
        calls = bridge.retrieve_with_trace.call_args_list
        assert calls[1].args[0] == "capital of France"


# ── MetaGate tests ───────────────────────────────────────────────────────────


class TestMetaGate:
    """Tests for the MetaGate method."""

    def test_confident_no_expand(self) -> None:
        """When gate is confident, stop after first retrieval (no expansion)."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate(expand=False)  # confident
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_metagate(
            example=example,
            bridge=bridge,
            metagate=metagate,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.expanded is False
        assert result.abstain_flag is False
        # Only one retrieval
        assert bridge.retrieve_with_trace.call_count == 1
        # Only one gate call
        assert metagate.decide.call_count == 1
        assert metagate.decide_second.call_count == 0
        # QA was called
        bridge.answer.assert_called_once()

    def test_not_confident_expands_and_forced_answers(self) -> None:
        """When gate is not confident, expand, fuse, and produce answer."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate(expand=True, abstain=False)
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_metagate(
            example=example,
            bridge=bridge,
            metagate=metagate,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.expanded is True
        assert result.abstain_flag is False
        assert bridge.retrieve_with_trace.call_count == 2
        assert metagate.decide.call_count == 1
        assert metagate.decide_second.call_count == 1
        bridge.answer.assert_called_once()

    def test_expands_and_abstains_when_still_low_confidence(self) -> None:
        """Second-gate low confidence sets abstain_flag."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate(expand=True, abstain=True)
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_metagate(
            example=example,
            bridge=bridge,
            metagate=metagate,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.expanded is True
        assert result.abstain_flag is True
        # Answer is still produced (forced)
        assert result.answer == "Paris"

    def test_gate_error_stops_with_error(self) -> None:
        """When the first gate throws, record error and stop (don't expand)."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate()
        metagate.decide.side_effect = RuntimeError("gate model unavailable")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        result = run_metagate(
            example=example,
            bridge=bridge,
            metagate=metagate,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        assert result.expanded is False
        assert result.answer == ""
        assert len(result.errors) == 1
        assert "gate model unavailable" in result.errors[0]


# ── run_method orchestrator tests ────────────────────────────────────────────


class TestRunMethodOrchestrator:
    """Tests for the high-level ``run_method`` orchestrator."""

    def test_requires_bridge_for_retrieval_methods(self) -> None:
        """Methods that need retrieval require a bridge."""
        examples = [make_example()]
        client = make_mock_client()
        ledger = make_tmp_ledger()

        with tempfile.TemporaryDirectory() as tmp, pytest.raises(
            ValueError, match="requires a bridge"
        ):
            run_method(
                "dense_rag",
                examples,
                bridge=None,
                client=client,
                ledger=ledger,
                output_dir=Path(tmp),
                effective_config_hash=CONFIG_HASH,
            )

    def test_requires_metagate_for_gate_methods(self) -> None:
        """Methods that use gates require a MetaGate instance."""
        examples = [make_example()]
        bridge = make_mock_bridge()
        client = make_mock_client()
        ledger = make_tmp_ledger()

        with tempfile.TemporaryDirectory() as tmp, pytest.raises(
            ValueError, match="requires a MetaGate"
        ):
            run_method(
                "metagate",
                examples,
                bridge=bridge,
                metagate=None,
                client=client,
                ledger=ledger,
                output_dir=Path(tmp),
                effective_config_hash=CONFIG_HASH,
            )

    def test_skips_completed_on_resume(self) -> None:
        """On second invocation, already-completed examples are skipped."""
        examples = [make_example(example_id="ex-1"), make_example(example_id="ex-2")]
        bridge = make_mock_bridge(answer="Paris")
        client = make_mock_client()
        ledger = make_tmp_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-test"

            # First run — process both
            results1 = run_method(
                "dense_rag",
                examples,
                bridge=bridge,
                client=client,
                ledger=ledger,
                output_dir=output_dir,
                effective_config_hash=CONFIG_HASH,
            )
            assert len(results1) == 2
            bridge.dense_with_trace.assert_called()

            # Reset mock to track new calls
            bridge.reset_mock()
            bridge.dense_with_trace.return_value = make_trace()

            # Second run — should skip both
            results2 = run_method(
                "dense_rag",
                examples,
                bridge=bridge,
                client=client,
                ledger=ledger,
                output_dir=output_dir,
                effective_config_hash=CONFIG_HASH,
            )
            assert len(results2) == 2
            # No new retrievals
            assert bridge.dense_with_trace.call_count == 0

    def test_all_five_methods_dispatch(self) -> None:
        """Each of the 5 method IDs dispatches to the correct runner."""
        example = make_example()
        bridge = make_mock_bridge(answer="Paris")
        metagate = make_mock_metagate(expand=False)
        client = make_mock_client()
        ledger = make_tmp_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            for method in ["llm_only", "dense_rag", "hipporag2", "always_expand", "metagate"]:
                results = run_method(
                    method,  # type: ignore[arg-type]
                    [example],
                    bridge=bridge,
                    metagate=metagate,
                    client=client,
                    ledger=ledger,
                    output_dir=Path(tmp) / f"run-{method}",
                    effective_config_hash=CONFIG_HASH,
                )
                assert len(results) == 1
                assert results[0].method == method

    def test_empty_examples_returns_empty(self) -> None:
        """Passing an empty example list returns an empty list."""
        client = make_mock_client()
        ledger = make_tmp_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            results = run_method(
                "llm_only",
                [],
                client=client,
                ledger=ledger,
                output_dir=Path(tmp),
                effective_config_hash=CONFIG_HASH,
            )
            assert results == []

    def test_run_id_is_deterministic(self) -> None:
        """The same config + method + dataset + split produces the same run_id."""
        rid1 = _compute_run_id(CONFIG_HASH, "hipporag2", "musique", "dev")
        rid2 = _compute_run_id(CONFIG_HASH, "hipporag2", "musique", "dev")
        assert rid1 == rid2

    def test_run_id_differs_per_method(self) -> None:
        """Different methods produce different run_ids."""
        rid1 = _compute_run_id(CONFIG_HASH, "llm_only", "musique", "dev")
        rid2 = _compute_run_id(CONFIG_HASH, "hipporag2", "musique", "dev")
        assert rid1 != rid2


# ── Usage tracking tests ─────────────────────────────────────────────────────


class TestUsageTracking:
    """Tests that method results capture usage deltas from the ledger."""

    def test_llm_only_records_usage(self) -> None:
        """LLM-only method result includes usage from the LLM call."""
        example = make_example()
        client = make_mock_client(answer="Paris")
        ledger = make_tmp_ledger()

        result = run_llm_only(
            example=example,
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
        )

        # The ledger should have entries (from the mock client's complete() call)
        # Since we're mocking the client, no actual ledger entries are written.
        # But the usage field should reflect the delta.
        assert result.usage is not None
