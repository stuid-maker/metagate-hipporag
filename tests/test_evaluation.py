"""Tests for evaluation module — retrieval, QA, calibration, coverage, risk, and gate-efficiency metrics.

Hand-computed small examples verify that Recall, EM, F1, Brier, and ECE
match reference calculations (spec §11.1).
"""

from __future__ import annotations

import math

import pytest

from metagate_hipporag.evaluation import (
    EvaluationReport,
    compute_brier,
    compute_ece,
    compute_em,
    compute_f1,
    compute_recall,
    evaluate,
    normalize_answer,
)
from metagate_hipporag.models import (
    Example,
    GateDecision,
    MethodResult,
    RetrievalTrace,
    RetrievedPassage,
    Usage,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _ex(dataset: str = "nq_rear", example_id: str = "0") -> Example:
    return Example(
        dataset="nq_rear",
        example_id=example_id,
        question="Who is Alice?",
        gold_answers=["alice"],
        gold_docs=["Doc1\ncontent one"],
        stratum="simple",
    )


def _passage(chunk_id: str, text: str, score: float = 0.9, rank: int = 1) -> RetrievedPassage:
    return RetrievedPassage(chunk_id=chunk_id, text=text, score=score, rank=rank)


def _trace(query: str, passages: list[RetrievedPassage]) -> RetrievalTrace:
    return RetrievalTrace(
        retrieval_query=query,
        passages=passages,
        facts_before_filter=[],
        facts_after_filter=[],
        used_dense_fallback=False,
    )


def _gate(prob: float) -> GateDecision:
    return GateDecision(
        evidence_sufficient_probability=prob,
        missing_information="test",
        retrieval_rewrite="What about Alice?",
        rationale_summary="test gate call",
    )


# ── normalize_answer ──────────────────────────────────────────────────────────


class TestNormalizeAnswer:
    def test_lowercase_and_strip(self) -> None:
        assert normalize_answer("  Hello World  ") == "hello world"

    def test_removes_punctuation(self) -> None:
        assert normalize_answer("Hello, world!") == "hello world"

    def test_removes_articles(self) -> None:
        assert normalize_answer("The answer is a test") == "answer is test"

    def test_collapse_whitespace(self) -> None:
        assert normalize_answer("foo   bar\n\tbaz") == "foo bar baz"

    def test_empty_string(self) -> None:
        assert normalize_answer("") == ""

    def test_only_articles(self) -> None:
        assert normalize_answer("a an the") == ""


# ── compute_em ────────────────────────────────────────────────────────────────


class TestComputeEM:
    def test_exact_match(self) -> None:
        assert compute_em("Alice", ["alice"]) == 1.0

    def test_no_match(self) -> None:
        assert compute_em("Bob", ["alice"]) == 0.0

    def test_multiple_gold_max(self) -> None:
        # "alice" normalizes to "alice"; "bob" stays "bob"
        assert compute_em("bob", ["alice", "bob"]) == 1.0

    def test_empty_prediction(self) -> None:
        assert compute_em("", ["alice"]) == 0.0

    def test_normalization_effect(self) -> None:
        assert compute_em("The  Alice!", ["  Alice  "]) == 1.0


# ── compute_f1 ────────────────────────────────────────────────────────────────


class TestComputeF1:
    def test_perfect_f1(self) -> None:
        assert compute_f1("alice was here", ["alice was here"]) == 1.0

    def test_zero_f1(self) -> None:
        assert compute_f1("bob", ["alice"]) == 0.0

    def test_partial_overlap(self) -> None:
        # pred: "foo" → tokens=["foo"]; gold: "foo bar" → tokens=["foo","bar"]
        # common=1; P=1/1=1.0, R=1/2=0.5; F1=2*1*0.5/1.5=0.666...
        f1 = compute_f1("foo", ["foo bar"])
        assert math.isclose(f1, 2 / 3, rel_tol=1e-9)

    def test_multiple_gold_max(self) -> None:
        # pred "x y" vs gold ["x y z", "w"]:
        # vs gold1: tokens pred=[x,y], gold=[x,y,z]; common=[x,y]=2; P=1.0, R=2/3, F1=0.8
        # vs gold2: F1=0 (no overlap)
        # max = 0.8
        f1 = compute_f1("x y", ["x y z", "w"])
        assert math.isclose(f1, 0.8, rel_tol=1e-9)

    def test_empty_prediction(self) -> None:
        assert compute_f1("", ["alice"]) == 0.0

    def test_empty_gold(self) -> None:
        with pytest.raises(ValueError):
            compute_f1("x", [])


# ── compute_recall ────────────────────────────────────────────────────────────


class TestComputeRecall:
    def test_full_recall(self) -> None:
        passages = [_passage("a1", "Doc1\ntext"), _passage("a2", "Doc2\ntext")]
        gold = ["Doc1\ntext"]
        result = compute_recall(passages, gold)
        assert result["Recall@2"] == 1.0
        assert result["Recall@5"] == 1.0

    def test_partial_recall(self) -> None:
        passages = [_passage("a1", "Doc3\ntext")]
        gold = ["Doc1\ntext", "Doc2\ntext"]
        result = compute_recall(passages, gold)
        assert result["Recall@2"] == 0.0
        assert result["Recall@5"] == 0.0

    def test_half_recall(self) -> None:
        passages = [_passage("a1", "Doc1\ntext")]
        gold = ["Doc1\ntext", "Doc2\ntext"]
        result = compute_recall(passages, gold)
        assert result["Recall@2"] == 0.5
        assert result["Recall@5"] == 0.5

    def test_truncation_at_k(self) -> None:
        passages = [
            _passage("a1", "Doc3\ntext"),
            _passage("a2", "Doc4\ntext"),
            _passage("a3", "Doc1\ntext"),
        ]
        gold = ["Doc1\ntext", "Doc2\ntext"]
        result = compute_recall(passages, gold)
        # Recall@2 looks at first 2 passages [Doc3, Doc4] → 0 match → 0.0
        # Recall@5 looks at all 3 → 1 match → 0.5
        assert result["Recall@2"] == 0.0
        assert result["Recall@5"] == 0.5

    def test_empty_passages(self) -> None:
        result = compute_recall([], ["Doc1\ntext"])
        assert result["Recall@2"] == 0.0
        assert result["Recall@5"] == 0.0

    def test_empty_gold_docs(self) -> None:
        passages = [_passage("a1", "Doc1\ntext")]
        result = compute_recall(passages, [])
        assert result["Recall@2"] == 0.0
        assert result["Recall@5"] == 0.0

    def test_deduplicated_passages(self) -> None:
        """Duplicated passage texts count as one for recall."""
        passages = [
            _passage("a1", "Doc1\ntext"),
            _passage("a2", "Doc1\ntext"),
        ]
        gold = ["Doc1\ntext"]
        result = compute_recall(passages, gold)
        # Only one unique gold doc retrieved
        assert result["Recall@2"] == 1.0


# ── Brier and ECE ─────────────────────────────────────────────────────────────


class TestBrier:
    def test_perfect_calibration(self) -> None:
        brier = compute_brier([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0])
        # (0.1^2 + 0.2^2 + 0.1^2 + 0.2^2)/4 = (0.01+0.04+0.01+0.04)/4 = 0.025
        assert math.isclose(brier, 0.025, rel_tol=1e-9)

    def test_worst_calibration(self) -> None:
        brier = compute_brier([0.9, 0.8], [0, 0])
        # (0.9^2 + 0.8^2)/2 = (0.81+0.64)/2 = 0.725
        assert math.isclose(brier, 0.725, rel_tol=1e-9)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            compute_brier([], [])


class TestECE:
    def test_well_calibrated(self) -> None:
        """6 samples, 3 bins. Data is well-separated but bin confidences
        don't perfectly match accuracies — ECE ≈ 0.067 (not zero)."""
        probs = [0.95, 0.85, 0.55, 0.45, 0.15, 0.05]
        labels = [1, 1, 1, 0, 0, 0]
        # Sorted: [0.05(l=0),0.15(l=0),0.45(l=0),0.55(l=1),0.85(l=1),0.95(l=1)]
        # Bin0: [0.05,0.15] → acc=0, conf=0.1 → diff=0.1
        # Bin1: [0.45,0.55] → acc=0.5, conf=0.5 → diff=0.0
        # Bin2: [0.85,0.95] → acc=1, conf=0.9 → diff=0.1
        # weighted = (2/6)*0.1 + (2/6)*0.0 + (2/6)*0.1 = 0.0667
        ece = compute_ece(probs, labels, n_bins=3)
        assert math.isclose(ece, 0.0666667, abs_tol=1e-6)

    def test_miscalibrated(self) -> None:
        """3 samples, 3 bins: one sample per bin."""
        probs = [0.9, 0.5, 0.1]
        labels = [1, 0, 0]
        # Sorted: [0.1(l=0), 0.5(l=0), 0.9(l=1)]
        # Bin0: prob=0.1, label=0 → diff=0.1
        # Bin1: prob=0.5, label=0 → diff=0.5
        # Bin2: prob=0.9, label=1 → diff=0.1
        # ECE = (1/3)*0.1 + (1/3)*0.5 + (1/3)*0.1 = 0.23333
        ece = compute_ece(probs, labels, n_bins=3)
        assert math.isclose(ece, 0.2333333, rel_tol=1e-6)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            compute_ece([], [], n_bins=3)


# ── Hand-computed 4-example metagate scenario ─────────────────────────────────
# Threshold = 0.50
#
# Example A (expand=False, correct, R@5=1, gate prob=0.95):
#   first passages = [Doc1, Doc2]  → gold=[Doc1]  → R@5=1.0, sufficient.
#   gate prob=0.95 >= 0.50 → stop.  Answer="alice" → EM=1, F1=1.
#   Gate decisions=[g0(prob=0.95)].
#
# Example B (expand=False, false-stop, R@5=0, gate prob=0.92):
#   first passages = [Doc3]  → gold=[Doc1]  → R@5=0.0, insufficient.
#   gate prob=0.92 >= 0.50 → stop (false!).  Answer="bob" → EM=0, F1=0.
#   Gate decisions=[g0(prob=0.92)].
#
# Example C (expand=True, R@5_first=0 → R@5_fused=1, gate prob_first=0.30):
#   first passages = [Doc3]  → R@5=0.0, insufficient.
#   gate prob=0.30 < 0.50 → expand.
#   second retrieval = [Doc1, Doc4].  RRF fused = [Doc1, Doc3, Doc4].
#   gold=[Doc1] → R@5_fused=1.0.  Answer="alice" → EM=1, F1=1.
#   Gate decisions=[g0(prob=0.30), g1(prob=0.85)].
#
# Example D (expand=True, unnecessary, R@5_first=1, gate prob_first=0.40):
#   first passages = [Doc1, Doc2] → R@5=1.0, sufficient.
#   gate prob=0.40 < 0.50 → expand unnecessarily.
#   second = [Doc5].  Fused = [Doc1, Doc2, Doc5].  R@5_fused=1.0.
#   Answer="alice" → EM=1, F1=1.
#   Gate decisions=[g0(prob=0.40), g1(prob=0.88)].


def _build_metagate_results() -> list[MethodResult]:
    """Construct 4 hand-computed metagate MethodResults."""
    run_id = "test-run"
    ex = _ex

    # --- Example A: confident stop, correct ---
    first_a = _trace("Who is Alice?", [
        _passage("a1", "Doc1\ncontent one", rank=1),
        _passage("a2", "Doc2\nother", rank=2),
    ])
    result_a = MethodResult(
        run_id=run_id,
        method="metagate",
        example=ex("nq_rear", "A"),
        first_retrieval=first_a,
        second_retrieval=None,
        fused_passages=first_a.passages,
        answer="alice",
        gate_decisions=[_gate(0.95)],
        expanded=False,
        abstain_flag=False,
        usage=Usage(
            prompt_tokens=200,
            completion_tokens=50,
            embedding_tokens=1200,
            observed_latency_seconds=1.5,
            method_equivalent_latency_seconds=1.5,
            actual_usd=0.0003,
            method_equivalent_usd=0.0003,
        ),
        errors=[],
    )

    # --- Example B: confident false-stop, wrong ---
    first_b = _trace("Who is Alice?", [
        _passage("b1", "Doc3\nwrong", rank=1),
    ])
    result_b = MethodResult(
        run_id=run_id,
        method="metagate",
        example=ex("nq_rear", "B"),
        first_retrieval=first_b,
        second_retrieval=None,
        fused_passages=first_b.passages,
        answer="bob",
        gate_decisions=[_gate(0.92)],
        expanded=False,
        abstain_flag=False,
        usage=Usage(
            prompt_tokens=180,
            completion_tokens=30,
            embedding_tokens=1200,
            observed_latency_seconds=1.2,
            actual_usd=0.0002,
            method_equivalent_usd=0.0002,
        ),
        errors=[],
    )

    # --- Example C: expand, first insufficient, fixes it ---
    first_c = _trace("Who is Alice?", [
        _passage("c1", "Doc3\nwrong", rank=1),
    ])
    second_c = _trace("What about Alice?", [
        _passage("c2", "Doc1\ncontent one", rank=1),
        _passage("c3", "Doc4\nextra", rank=2),
    ])
    fused_c = [
        _passage("c2", "Doc1\ncontent one", rank=1),
        _passage("c1", "Doc3\nwrong", rank=2),
        _passage("c3", "Doc4\nextra", rank=3),
    ]
    result_c = MethodResult(
        run_id=run_id,
        method="metagate",
        example=ex("nq_rear", "C"),
        first_retrieval=first_c,
        second_retrieval=second_c,
        fused_passages=fused_c,
        answer="alice",
        gate_decisions=[_gate(0.30), _gate(0.85)],
        expanded=True,
        abstain_flag=False,  # second gate prob 0.85 >= 0.50, no abstain
        usage=Usage(
            prompt_tokens=500,
            completion_tokens=100,
            embedding_tokens=2400,
            observed_latency_seconds=3.0,
            actual_usd=0.0008,
            method_equivalent_usd=0.0008,
        ),
        errors=[],
    )

    # --- Example D: expand, first sufficient (unnecessary), answer still correct ---
    first_d = _trace("Who is Alice?", [
        _passage("d1", "Doc1\ncontent one", rank=1),
        _passage("d2", "Doc2\nother", rank=2),
    ])
    second_d = _trace("What about Alice?", [
        _passage("d3", "Doc5\nextra", rank=1),
    ])
    fused_d = [
        _passage("d1", "Doc1\ncontent one", rank=1),
        _passage("d2", "Doc2\nother", rank=2),
        _passage("d3", "Doc5\nextra", rank=3),
    ]
    result_d = MethodResult(
        run_id=run_id,
        method="metagate",
        example=ex("nq_rear", "D"),
        first_retrieval=first_d,
        second_retrieval=second_d,
        fused_passages=fused_d,
        answer="alice",
        gate_decisions=[_gate(0.40), _gate(0.88)],
        expanded=True,
        abstain_flag=False,
        usage=Usage(
            prompt_tokens=500,
            completion_tokens=100,
            embedding_tokens=2400,
            observed_latency_seconds=3.0,
            actual_usd=0.0007,
            method_equivalent_usd=0.0007,
        ),
        errors=[],
    )

    return [result_a, result_b, result_c, result_d]


EXPECTED_FUSED_RECALL_AT_2: float = (
    1.0  # A
    + 0.0  # B
    + 1.0  # C (Doc1 is in top 2 of fused)
    + 1.0  # D (Doc1 is in top 2 of fused)
) / 4  # = 0.75

EXPECTED_FUSED_RECALL_AT_5: float = (
    1.0  # A
    + 0.0  # B
    + 1.0  # C
    + 1.0  # D
) / 4  # = 0.75

EXPECTED_FIRST_RECALL_AT_2: float = (
    1.0  # A: Doc1 in first 2 → R@2=1
    + 0.0  # B: Doc3 → R@2=0
    + 0.0  # C: Doc3 → R@2=0
    + 1.0  # D: Doc1 in first 2 → R@2=1
) / 4  # = 0.50

EXPECTED_FIRST_RECALL_AT_5: float = EXPECTED_FIRST_RECALL_AT_2  # same, all <5 passages

EXPECTED_EM: float = (1.0 + 0.0 + 1.0 + 1.0) / 4  # = 0.75
EXPECTED_F1: float = (1.0 + 0.0 + 1.0 + 1.0) / 4  # = 0.75

# Brier: probs=[0.95, 0.92, 0.30, 0.40], labels=[1, 0, 0, 1]
# Brier = ((0.95-1)^2 + (0.92-0)^2 + (0.30-0)^2 + (0.40-1)^2)/4
#       = (0.0025 + 0.8464 + 0.09 + 0.36) / 4 = 1.2989 / 4 = 0.324725
EXPECTED_BRIER: float = 0.324725

# ECE (10 equal-freq bins for 4 samples): binary search sorted probs:
# Sorted: [(0.30,l=0), (0.40,l=1), (0.92,l=0), (0.95,l=1)]
# 4 samples into 10 bins → each bin has at most 1 sample (some bins empty)
# This is a degenerate case; with 4 samples and 10 bins, ECE is effectively
# per-sample error average. Let's use n_bins=2 for the hand-computed test.
# Sorted: [0.30, 0.40, 0.92, 0.95]; labels: [0, 1, 0, 1]
# 2 bins, 2 samples each:
# Bin 0: probs [0.30, 0.40], labels [0, 1]; avg_prob=0.35, avg_label=0.5
# Bin 1: probs [0.92, 0.95], labels [0, 1]; avg_prob=0.935, avg_label=0.5
# ECE_2 = (|0.35-0.5| + |0.935-0.5|) * (2/4) / 2? No.
# Standard ECE: sum_i (|B_i|/N) * |avg_prob_i - avg_label_i|
# = (2/4)*|0.35-0.5| + (2/4)*|0.935-0.5| = 0.5*0.15 + 0.5*0.435 = 0.075 + 0.2175 = 0.2925
EXPECTED_ECE_2: float = 0.2925

# False stop rate (gate prob >= 0.50):
# A: prob=0.95, R@5_first=1 → sufficient, no false stop
# B: prob=0.92, R@5_first=0 → insufficient, false stop ✓ (numerator)
# C: prob=0.30, R@5_first=0 → insufficient, expanded (denominator)
# D: prob=0.40, R@5_first=1 → sufficient, expanded (denominator for unnecessary)
# Denominator (insufficient): B + C = 2; False stop: B = 1 → rate = 0.5
EXPECTED_FALSE_STOP_RATE: float = 0.5
EXPECTED_FALSE_STOP_COUNT: int = 1
EXPECTED_INSUFFICIENT_COUNT: int = 2

# Unnecessary expansion rate:
# Denominator (sufficient): A + D = 2; Expanded: D = 1 → rate = 0.5
EXPECTED_UNNECESSARY_EXPANSION_RATE: float = 0.5
EXPECTED_UNNECESSARY_COUNT: int = 1
EXPECTED_SUFFICIENT_COUNT: int = 2

# Gate-Efficiency:
# Stop correctness: gate said stop for A, B (2). Actually sufficient: A only (1). Rate = 1/2 = 0.5
# Expand benefit: gate said expand for C, D (2). 
#   C: first EM=0 (first passages [Doc3] → answer would be wrong), fused EM=1 → benefit ✓
#   D: first EM=1, fused EM=1 → no net EM gain from expansion. Benefit=1/2=0.5
# ΔRecall expanded: C: R@5 0→1 = +1; D: R@5 already 1→1 = 0. Mean=0.5
# Benefit = expanded AND recall improved (C only: 1/2)
EXPECTED_STOP_CORRECTNESS: float = 0.5
EXPECTED_EXPAND_BENEFIT: float = 0.5
EXPECTED_DELTA_RECALL_EXPAND: float = 0.5


class TestEvaluateMetagateHandComputed:
    """Canonical hand-computed 4-example metagate scenario."""

    @pytest.fixture
    def results(self) -> list[MethodResult]:
        return _build_metagate_results()

    def test_fused_recall(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(report.recall_at_2, EXPECTED_FUSED_RECALL_AT_2, rel_tol=1e-9)
        assert math.isclose(report.recall_at_5, EXPECTED_FUSED_RECALL_AT_5, rel_tol=1e-9)

    def test_first_recall(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(
            report.first_recall_at_5, EXPECTED_FIRST_RECALL_AT_5, rel_tol=1e-9
        )

    def test_em_f1(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(report.em, EXPECTED_EM, rel_tol=1e-9)
        assert math.isclose(report.token_f1, EXPECTED_F1, rel_tol=1e-9)

    def test_brier(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(report.brier, EXPECTED_BRIER, rel_tol=1e-9)

    def test_ece(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(report.ece, EXPECTED_ECE_2, rel_tol=1e-9)

    def test_false_stop_rate(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(
            report.false_stop_rate_insufficient, EXPECTED_FALSE_STOP_RATE, rel_tol=1e-9
        )
        assert report.false_stop_count == EXPECTED_FALSE_STOP_COUNT
        assert report.first_round_insufficient_count == EXPECTED_INSUFFICIENT_COUNT

    def test_unnecessary_expansion_rate(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        assert math.isclose(
            report.unnecessary_expansion_rate_sufficient,
            EXPECTED_UNNECESSARY_EXPANSION_RATE,
            rel_tol=1e-9,
        )
        assert report.unnecessary_expansion_count == EXPECTED_UNNECESSARY_COUNT
        assert report.first_round_sufficient_count == EXPECTED_SUFFICIENT_COUNT

    def test_gate_efficiency(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        ge = report.gate_efficiency
        assert ge is not None
        assert math.isclose(ge.stop_correctness, EXPECTED_STOP_CORRECTNESS, rel_tol=1e-9)
        assert math.isclose(ge.expand_benefit_rate, EXPECTED_EXPAND_BENEFIT, rel_tol=1e-9)
        assert math.isclose(
            ge.delta_recall_expand, EXPECTED_DELTA_RECALL_EXPAND, rel_tol=1e-9
        )

    def test_expansion_rate(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        # C and D expanded: 2/4 = 0.5
        assert math.isclose(report.expansion_rate, 0.5, rel_tol=1e-9)

    def test_coverage_risk(self, results: list[MethodResult]) -> None:
        report = evaluate(results, gate_threshold=0.50, ece_bins=2)
        # All 4 are forced-answer → coverage = 1.0
        # EM among forced: 3/4 = 0.75, risk = 0.25
        assert math.isclose(report.coverage, 1.0, rel_tol=1e-9)
        assert math.isclose(report.selective_risk_em, 1.0 - EXPECTED_EM, rel_tol=1e-9)


# ── LLM-only method (no retrieval, Recall=N/A) ───────────────────────────────


class TestEvaluateLLMOnly:
    def test_recall_is_na(self) -> None:
        results = [
            MethodResult(
                run_id="r",
                method="llm_only",
                example=_ex("nq_rear", "0"),
                first_retrieval=None,
                second_retrieval=None,
                fused_passages=[],
                answer="alice",
                gate_decisions=[],
                expanded=False,
                abstain_flag=False,
                usage=Usage(prompt_tokens=50, completion_tokens=5),
                errors=[],
            )
        ]
        report = evaluate(results, gate_threshold=0.50)
        assert report.recall_at_2 is None
        assert report.recall_at_5 is None
        assert report.em == 1.0
        assert report.token_f1 == 1.0


# ── No-gate methods skip calibration ──────────────────────────────────────────


class TestEvaluateNoGate:
    def test_hipporag2_skips_gate_metrics(self) -> None:
        passages = [_passage("a1", "Doc1\ncontent one")]
        results = [
            MethodResult(
                run_id="r",
                method="hipporag2",
                example=_ex("nq_rear", "0"),
                first_retrieval=_trace("q", passages),
                second_retrieval=None,
                fused_passages=passages,
                answer="alice",
                gate_decisions=[],
                expanded=False,
                abstain_flag=False,
                usage=Usage(),
                errors=[],
            )
        ]
        report = evaluate(results, gate_threshold=0.50)
        assert report.recall_at_5 == 1.0
        assert report.em == 1.0
        assert report.brier is None  # no gate data
        assert report.ece is None
        assert report.false_stop_rate_insufficient is None
        assert report.gate_efficiency is None


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_results(self) -> None:
        with pytest.raises(ValueError):
            evaluate([], gate_threshold=0.50)

    def test_abstain_flag(self) -> None:
        """Example with abstain_flag=True: coverage < 1, selective risk based on forced answers."""
        passages = [_passage("a1", "Doc3\nwrong")]
        results = [
            MethodResult(
                run_id="r",
                method="metagate",
                example=_ex("nq_rear", "0"),
                first_retrieval=_trace("q", passages),
                second_retrieval=_trace("q2", [_passage("a2", "Doc3\nwrong")]),
                fused_passages=passages,
                answer="bob",  # forced answer, wrong
                gate_decisions=[_gate(0.30), _gate(0.10)],
                expanded=True,
                abstain_flag=True,
                usage=Usage(),
                errors=[],
            )
        ]
        report = evaluate(results, gate_threshold=0.50)
        assert report.coverage == 0.0  # abstain → not covered
        # No forced answers → selective risk is undefined (NaN)
        assert math.isnan(report.selective_risk_em)
        assert report.forced_em == 0.0

    def test_cache_hit_and_cost(self) -> None:
        passages = [_passage("a1", "Doc1\ncontent one")]
        results = [
            MethodResult(
                run_id="r",
                method="dense_rag",
                example=_ex("nq_rear", "0"),
                first_retrieval=_trace("q", passages),
                second_retrieval=None,
                fused_passages=passages,
                answer="alice",
                gate_decisions=[],
                expanded=False,
                abstain_flag=False,
                usage=Usage(
                    prompt_tokens=200,
                    completion_tokens=50,
                    embedding_tokens=300,
                    observed_latency_seconds=0.8,
                    actual_usd=0.0002,
                    method_equivalent_usd=0.0003,
                ),
                errors=[],
            )
        ]
        report = evaluate(results, gate_threshold=0.50)
        assert report.total_prompt_tokens == 200
        assert report.total_embedding_tokens == 300
        assert report.actual_cost_usd == 0.0002
        assert report.method_equivalent_cost_usd == 0.0003
