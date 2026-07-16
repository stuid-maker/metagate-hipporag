"""Tests for MetaGate — zero-shot gate, threshold selection, and expansion policy.

Covers:
- ``select_threshold`` — balanced accuracy with tie-breakers
- ``GateDecision`` model validation (bounded probability, non-empty rewrite)
- ``MetaGate`` controller (decision flow, expansion policy, abstain flag)
- Gate prompt format and content constraints
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from metagate_hipporag.metagate import (
    DEFAULT_GATE_PROMPT,
    MetaGate,
    build_first_gate_message,
    build_second_gate_message,
    gate_prompt_config,
    select_threshold,
)
from metagate_hipporag.models import (
    GateDecision,
    RetrievalTrace,
    RetrievedPassage,
)

# ── Threshold selection tests ────────────────────────────────────────────────


class TestSelectThreshold:
    """Tests for dev-only threshold selection."""

    def test_threshold_uses_balanced_accuracy_then_lower_expansion(self) -> None:
        """Plan's canonical example: tie on balanced accuracy,
        broken by lower expansion rate, then higher threshold.
        """
        probabilities = [0.9, 0.8, 0.4, 0.2]
        sufficient = [True, True, False, False]
        assert (
            select_threshold(probabilities, sufficient, [0.5, 0.75, 0.85]) == 0.75
        )

    def test_threshold_selects_highest_balanced_accuracy(self) -> None:
        """When one candidate has strictly better balanced accuracy."""
        probs = [0.95, 0.6, 0.3]
        sufficient = [True, False, False]
        # At threshold 0.9: predict sufficient for prob>=0.9 → [T,F,F] → TP=1,TN=2 → BA=1.0
        # At threshold 0.7: predict sufficient for prob>=0.7 → [T,F,F] → same
        # At threshold 0.5: predict sufficient for prob>=0.5 → [T,T,F] → TP=1,TN=1,FP=1 → BA=0.75
        result = select_threshold(probs, sufficient, [0.5, 0.7, 0.9])
        assert result in (0.7, 0.9)  # both get perfect BA; tie-break to 0.7 or 0.9

    def test_threshold_all_sufficient(self) -> None:
        """All examples sufficient → any threshold works, picks highest."""
        probs = [0.7, 0.8, 0.9]
        sufficient = [True, True, True]
        result = select_threshold(probs, sufficient, [0.5, 0.75])
        # All get BA=1.0, tie-break: lower expansion rate → 0.75 has lower expansion
        # But for probs [0.7,0.8,0.9]: 0.5 expands 0%, 0.75 expands 33% → 0.5 has lower expansion
        assert result == 0.5

    def test_threshold_none_sufficient(self) -> None:
        """No examples sufficient → any threshold works; lowest expansion then highest threshold."""
        probs = [0.1, 0.2, 0.3]
        sufficient = [False, False, False]
        result = select_threshold(probs, sufficient, [0.5, 0.7])
        # Both expand 0% (no sufficient cases to stop on), BA=1.0 for both
        # Tie-break: lower expansion → same → higher threshold → 0.7
        assert result == 0.7

    def test_threshold_rejects_empty_candidates(self) -> None:
        """Empty candidate list raises ValueError."""
        with pytest.raises(ValueError, match="threshold_candidates"):
            select_threshold([], [], [])

    def test_threshold_rejects_mismatched_lengths(self) -> None:
        """Probabilities and sufficient must have same length."""
        with pytest.raises(ValueError, match="must have the same length"):
            select_threshold([0.5], [True, False], [0.5])


# ── Gate prompt tests ────────────────────────────────────────────────────────


class TestGatePrompt:
    """Tests for the frozen zero-shot gate prompt."""

    def test_gate_prompt_file_exists_and_is_valid_json(self) -> None:
        """Gate prompt config file must exist and be valid JSON."""
        assert gate_prompt_config().exists()
        data = json.loads(gate_prompt_config().read_text(encoding="utf-8"))
        assert "system_prompt" in data
        assert "sha256" in data
        assert len(data["sha256"]) == 64

    def test_gate_prompt_sha256_matches_content(self) -> None:
        """The stored SHA-256 must match the actual system prompt."""
        import hashlib

        data = json.loads(gate_prompt_config().read_text(encoding="utf-8"))
        prompt = data["system_prompt"]
        actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert data["sha256"] == actual

    def test_gate_prompt_contains_required_phrases(self) -> None:
        """Prompt must include key instructions from the spec."""
        data = json.loads(gate_prompt_config().read_text(encoding="utf-8"))
        prompt = data["system_prompt"]
        assert "evidence-sufficiency monitor" in prompt
        assert "Do not answer the question" in prompt
        assert "Ignore your parametric knowledge" in prompt
        assert "probability from 0 to 1" in prompt
        assert "missing information" in prompt
        assert "retrieval query" in prompt
        assert "rationale summary" in prompt

    def test_default_gate_prompt_matches_file(self) -> None:
        """DEFAULT_GATE_PROMPT constant matches the config file."""
        data = json.loads(gate_prompt_config().read_text(encoding="utf-8"))
        assert data["system_prompt"] == DEFAULT_GATE_PROMPT


# ── Gate message construction tests ──────────────────────────────────────────


class TestGateMessages:
    """Tests for first-gate and second-gate user message serialization."""

    def _make_trace(self, query: str, passages: list[RetrievedPassage]) -> RetrievalTrace:
        return RetrievalTrace(
            retrieval_query=query,
            passages=passages,
            facts_before_filter=[("s", "p", "o")],
            facts_after_filter=[("s", "p", "o")],
            used_dense_fallback=False,
        )

    def test_first_gate_message_includes_question_and_passages(self) -> None:
        """First-gate message must contain the original question and passages."""
        passages = [
            RetrievedPassage(chunk_id="c1", text="Passage 1", score=0.9, rank=1),
            RetrievedPassage(chunk_id="c2", text="Passage 2", score=0.8, rank=2),
        ]
        trace = self._make_trace("What is AI?", passages)
        messages = build_first_gate_message(trace)

        # messages is list[dict]; extract user content
        user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
        assert "What is AI?" in user_content
        assert "Passage 1" in user_content
        assert "Passage 2" in user_content
        assert "fact" in user_content.lower()

    def test_first_gate_message_excludes_dataset_and_gold(self) -> None:
        """First-gate message must NOT contain dataset name, gold answer, or method name."""
        passages = [RetrievedPassage(chunk_id="c1", text="Some text", score=0.5, rank=1)]
        trace = self._make_trace("Query", passages)
        messages = build_first_gate_message(trace)

        user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
        assert "musique" not in user_content.lower()
        assert "nq_rear" not in user_content.lower()
        assert "2wikimultihopqa" not in user_content.lower()
        assert "gold" not in user_content.lower()
        assert "metagate" not in user_content.lower()

    def test_second_gate_message_includes_both_queries(self) -> None:
        """Second-gate message must include both the original and rewrite queries."""
        passages = [
            RetrievedPassage(chunk_id="c1", text="Fused passage", score=0.7, rank=1),
        ]
        first_trace = self._make_trace("original query", [
            RetrievedPassage(chunk_id="a", text="A", score=0.5, rank=1),
        ])
        second_trace = self._make_trace("rewrite query", passages)
        messages = build_second_gate_message(first_trace, second_trace, passages)

        user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
        assert "original query" in user_content
        assert "rewrite query" in user_content
        assert "Fused passage" in user_content

    def test_second_gate_message_excludes_dataset_and_gold(self) -> None:
        """Second-gate message must NOT leak dataset or gold info."""
        passages = [RetrievedPassage(chunk_id="c1", text="Text", score=0.5, rank=1)]
        t1 = self._make_trace("Q1", passages)
        t2 = self._make_trace("Q2", passages)
        messages = build_second_gate_message(t1, t2, passages)

        user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
        assert "musique" not in user_content.lower()
        assert "gold" not in user_content.lower()
        assert "metagate" not in user_content.lower()


# ── MetaGate controller tests ────────────────────────────────────────────────


class TestMetaGateController:
    """Tests for the MetaGate expansion policy."""

    def test_metagate_stops_when_probability_above_threshold(self) -> None:
        """When gate probability ≥ threshold, do not expand."""
        # Create a mock client that returns high probability
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.value = GateDecision(
            evidence_sufficient_probability=0.9,
            missing_information="none",
            retrieval_rewrite="find more about AI",
            rationale_summary="Evidence covers the question.",
        )
        mock_client.complete.return_value = mock_completion

        mg = MetaGate(client=mock_client, threshold=0.5, max_expansions=1)

        first_trace = self._make_trace(
            "What is AI?",
            [RetrievedPassage(chunk_id="c1", text="AI is...", score=0.9, rank=1)],
        )

        decision = mg.decide(first_trace)
        assert decision.expand is False
        assert decision.gate.evidence_sufficient_probability == 0.9

    def test_metagate_expands_when_probability_below_threshold(self) -> None:
        """When gate probability < threshold, signal expansion."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.value = GateDecision(
            evidence_sufficient_probability=0.3,
            missing_information="definition of intelligence",
            retrieval_rewrite="what is artificial intelligence definition",
            rationale_summary="Passages lack a clear definition.",
        )
        mock_client.complete.return_value = mock_completion

        mg = MetaGate(client=mock_client, threshold=0.75, max_expansions=1)

        first_trace = self._make_trace(
            "What is AI?",
            [RetrievedPassage(chunk_id="c1", text="AI history...", score=0.9, rank=1)],
        )

        decision = mg.decide(first_trace)
        assert decision.expand is True
        assert decision.gate.retrieval_rewrite != ""

    def test_metagate_rejects_expansion_beyond_max(self) -> None:
        """max_expansions must be 1; requesting beyond that raises ValueError."""
        with pytest.raises(ValueError, match="max_expansions"):
            MetaGate(client=MagicMock(), threshold=0.5, max_expansions=2)

    def test_metagate_second_gate_sets_abstain_when_below_threshold(self) -> None:
        """After expansion, second gate below threshold → abstain_flag=True."""
        mock_client = MagicMock()
        # First gate: below threshold → expand
        call1 = MagicMock()
        call1.value = GateDecision(
            evidence_sufficient_probability=0.3,
            missing_information="missing fact",
            retrieval_rewrite="rewritten query",
            rationale_summary="Need more.",
        )
        # Second gate: still below threshold → abstain
        call2 = MagicMock()
        call2.value = GateDecision(
            evidence_sufficient_probability=0.4,
            missing_information="still missing",
            retrieval_rewrite="another query",
            rationale_summary="Still insufficient.",
        )
        mock_client.complete.side_effect = [call1, call2]

        mg = MetaGate(client=mock_client, threshold=0.75, max_expansions=1)

        first_trace = self._make_trace(
            "Q", [RetrievedPassage(chunk_id="c1", text="T", score=0.5, rank=1)]
        )
        second_trace = self._make_trace(
            "rewritten query",
            [RetrievedPassage(chunk_id="c2", text="T2", score=0.5, rank=1)],
        )

        # First decision: should expand
        d1 = mg.decide(first_trace)
        assert d1.expand is True

        # Second decision: should set abstain
        fused = [RetrievedPassage(chunk_id="c2", text="T2", score=0.5, rank=1)]
        d2 = mg.decide_second(first_trace, second_trace, fused, d1)
        assert d2.expand is False  # max_expansions=1 limits to 1 expansion
        assert d2.abstain is True

    def test_metagate_second_gate_clears_abstain_when_above_threshold(self) -> None:
        """After expansion, second gate above threshold → abstain_flag=False."""
        mock_client = MagicMock()
        call1 = MagicMock()
        call1.value = GateDecision(
            evidence_sufficient_probability=0.3,
            missing_information="missing fact",
            retrieval_rewrite="rewritten query",
            rationale_summary="Need more.",
        )
        call2 = MagicMock()
        call2.value = GateDecision(
            evidence_sufficient_probability=0.85,
            missing_information="none",
            retrieval_rewrite="backup query",
            rationale_summary="Now sufficient.",
        )
        mock_client.complete.side_effect = [call1, call2]

        mg = MetaGate(client=mock_client, threshold=0.75, max_expansions=1)

        first_trace = self._make_trace(
            "Q", [RetrievedPassage(chunk_id="c1", text="T", score=0.5, rank=1)]
        )
        second_trace = self._make_trace(
            "rewritten query",
            [RetrievedPassage(chunk_id="c2", text="T2", score=0.5, rank=1)],
        )

        d1 = mg.decide(first_trace)
        assert d1.expand is True

        fused2 = [RetrievedPassage(chunk_id="c2", text="T2", score=0.5, rank=1)]
        d2 = mg.decide_second(first_trace, second_trace, fused2, d1)
        assert d2.abstain is False

    def _make_trace(self, query: str, passages: list[RetrievedPassage]) -> RetrievalTrace:
        return RetrievalTrace(
            retrieval_query=query,
            passages=passages,
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=False,
        )


# ── GateDecision model validation ────────────────────────────────────────────


class TestGateDecisionModel:
    """Tests for the GateDecision Pydantic model."""

    def test_probability_bounded_0_to_1(self) -> None:
        """Probability must be in [0, 1]."""
        GateDecision(
            evidence_sufficient_probability=0.0,
            missing_information="none",
            retrieval_rewrite="query",
            rationale_summary="ok",
        )
        GateDecision(
            evidence_sufficient_probability=1.0,
            missing_information="none",
            retrieval_rewrite="query",
            rationale_summary="ok",
        )
        with pytest.raises(ValueError):
            GateDecision(
                evidence_sufficient_probability=1.1,
                missing_information="x",
                retrieval_rewrite="y",
                rationale_summary="z",
            )
        with pytest.raises(ValueError):
            GateDecision(
                evidence_sufficient_probability=-0.1,
                missing_information="x",
                retrieval_rewrite="y",
                rationale_summary="z",
            )

    def test_retrieval_rewrite_must_not_be_empty(self) -> None:
        """retrieval_rewrite is auto-stripped; empty after strip → error."""
        with pytest.raises(ValueError, match="retrieval_rewrite"):
            GateDecision(
                evidence_sufficient_probability=0.5,
                missing_information="none",
                retrieval_rewrite="   ",
                rationale_summary="ok",
            )

    def test_retrieval_rewrite_is_stripped(self) -> None:
        """Whitespace around retrieval_rewrite is stripped."""
        decision = GateDecision(
            evidence_sufficient_probability=0.5,
            missing_information="none",
            retrieval_rewrite="  query with space  ",
            rationale_summary="ok",
        )
        assert decision.retrieval_rewrite == "query with space"
