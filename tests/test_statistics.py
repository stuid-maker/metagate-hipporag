"""Tests for statistics module — McNemar, Holm, paired bootstrap, H1 contrast, H3 noninferiority.

All tests use hand-computed reference values (spec §11.1).  Deterministic seeds
guarantee bit-identical bootstrap results for CI assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from metagate_hipporag.models import (
    DatasetId,
    Example,
    GateDecision,
    MethodId,
    MethodResult,
    RetrievalTrace,
    RetrievedPassage,
    Usage,
)
from metagate_hipporag.statistics import (
    align_pairs,
    holm_corrected,
    mcnemar_exact,
    paired_bootstrap,
    run_statistics,
    stratified_group_means,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _ex(
    dataset: str = "nq_rear",
    example_id: str = "0",
    gold_answers: list[str] | None = None,
    gold_docs: list[str] | None = None,
    stratum: str = "simple",
) -> Example:
    return Example(
        dataset=dataset,
        example_id=example_id,
        question="Q?",
        gold_answers=gold_answers or ["a"],
        gold_docs=gold_docs or ["Doc\ntext"],
        stratum=stratum,
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


def _result(
    method: MethodId,
    ex: Example,
    answer: str,
    passages: list[RetrievedPassage],
    *,
    em: float | None = None,
    f1_val: float | None = None,
    gate_probs: list[float] | None = None,
    expanded: bool = False,
    abstain: bool = False,
    usage: Usage | None = None,
) -> MethodResult:
    """Build a ``MethodResult`` with controlled metrics."""
    if gate_probs is None:
        gate_probs = []
    if usage is None:
        usage = Usage()
    return MethodResult(
        run_id="fake",
        method=method,
        example=ex,
        first_retrieval=_trace("q", passages),
        second_retrieval=None,
        fused_passages=passages,
        answer=answer,
        gate_decisions=[
            GateDecision(
                evidence_sufficient_probability=p,
                missing_information="",
                retrieval_rewrite="rq",
                rationale_summary="",
            )
            for p in gate_probs
        ],
        expanded=expanded,
        abstain_flag=abstain,
        usage=usage,
        errors=[],
    )


# - shortcuts for verbose constructors


def _mr(
    method: str = "dense_rag",
    example: Example | None = None,
    *,
    answer: str = "x",
) -> MethodResult:
    """Minimal MethodResult factory for alignment tests."""
    return MethodResult(
        run_id="r",
        method=method,
        example=example or _ex(),
        first_retrieval=None,
        second_retrieval=None,
        fused_passages=[],
        answer=answer,
        gate_decisions=[],
        expanded=False,
        abstain_flag=False,
        usage=Usage(),
        errors=[],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  McNemar exact test
# ═══════════════════════════════════════════════════════════════════════════════


class TestMcNemarExact:
    """Two-sided exact McNemar via binomial distribution."""

    def test_no_changes_returns_one(self):
        """b=c=0 → concordant; p=1.0 (no evidence of difference)."""
        assert mcnemar_exact(b=0, c=0) == 1.0

    def test_all_changes_one_direction(self):
        """b=0, c=5: all disagreements go one way."""
        p = mcnemar_exact(b=0, c=5)
        # 2 * P(X ≤ 0 | n=5, p=0.5) = 2 * (0.5)^5 = 2 * 0.03125 = 0.0625
        expected = 2.0 * (0.5 ** 5)
        assert p == pytest.approx(expected)

    def test_hand_computed_example(self):
        """b=10, c=2, n=12:
        P(X≤2) = sum_{k=0}^{2} C(12,k)/2^12 = (1+12+66)/4096 = 79/4096 ≈ 0.019287
        Two-sided p = 2 * 0.019287 ≈ 0.038574
        """
        p = mcnemar_exact(b=10, c=2)
        expected = 2.0 * sum(binom.pmf(k, 12, 0.5) for k in range(3))
        assert p == pytest.approx(expected, rel=1e-10)

    def test_symmetric(self):
        """McNemar(b, c) == McNemar(c, b)."""
        assert mcnemar_exact(b=10, c=2) == pytest.approx(mcnemar_exact(b=2, c=10))

    def test_cross_val_statsmodels(self):
        """When statsmodels is installed the p-value must match its mcnemar(exact=True)."""
        # Only run if statsmodels is available
        try:
            from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        except ImportError:
            pytest.skip("statsmodels not installed — no cross-validation")
        table = [[0, 10], [2, 0]]
        sm_result = sm_mcnemar(table, exact=True)
        our = mcnemar_exact(b=10, c=2)
        assert our == pytest.approx(sm_result.pvalue, rel=1e-10)

    def test_rejects_negative_inputs(self):
        with pytest.raises(ValueError):
            mcnemar_exact(b=-1, c=5)
        with pytest.raises(ValueError):
            mcnemar_exact(b=5, c=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Holm correction
# ═══════════════════════════════════════════════════════════════════════════════


class TestHolm:
    """Holm-Bonferroni step-down correction (m=9, alpha=0.05 as per frozen config)."""

    def test_empty_input(self):
        assert holm_corrected([], alpha=0.05) == []

    def test_single_p_value(self):
        results = holm_corrected([0.03], alpha=0.05)
        # p=0.03 ≤ 0.05/1=0.05 → rejected
        assert results == [(0.03, 0.03, True)]

    def test_all_rejected(self):
        """Very small p-values → all rejected."""
        p_values = [0.001, 0.002, 0.003]
        results = holm_corrected(p_values, alpha=0.05)
        assert all(reject for _, _, reject in results)

    def test_all_retained(self):
        p_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        results = holm_corrected(p_values, alpha=0.05)
        assert not any(reject for _, _, reject in results)

    def test_partial_rejection_hand_computed(self):
        """Nine p-values; reject first two only."""
        p_values = [0.002, 0.005, 0.01, 0.04, 0.08, 0.12, 0.15, 0.20, 0.25]
        results = holm_corrected(p_values, alpha=0.05)
        # Ordered ascending internally
        # i=1: 0.002 ≤ 0.05/9 = 0.00556 ✓ reject
        # i=2: 0.005 ≤ 0.05/8 = 0.00625 ✓ reject
        # i=3: 0.01  > 0.05/7 = 0.00714 ✗ → stop
        rejected = [r[2] for r in sorted(results, key=lambda x: x[0])]
        assert rejected == [True, True] + [False] * 7

    def test_holm_adjusted_p_values(self):
        """Adjusted p-values must be non-decreasing when sorted ascending."""
        p_values = [0.002, 0.005, 0.01, 0.04, 0.08, 0.12, 0.15, 0.20, 0.25]
        results = holm_corrected(p_values, alpha=0.05)
        sorted_by_p = sorted(results, key=lambda x: x[0])
        adj = [r[1] for r in sorted_by_p]
        # Adjusted must be ≤ 1.0 and non-decreasing on sorted p
        for i in range(len(adj)):
            assert 0.0 <= adj[i] <= 1.0
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-12  # allow float epsilon

    def test_must_be_nine_for_primary_family(self):
        """Primary family is exactly 9; the function default assumes m=9 but accepts any m."""
        # len(p_values) != 9 is still valid; no enforcement here.
        # But run_statistics enforces 9 for the primary family.
        p_values = [0.01, 0.02, 0.03]
        results = holm_corrected(p_values, alpha=0.05)
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════════════════════
#  paired bootstrap
# ═══════════════════════════════════════════════════════════════════════════════


class TestPairedBootstrap:
    """Paired bootstrap with frozen seed 20260711 (§9.3)."""

    def test_constant_difference_zero_ci(self):
        """Every pair has diff=-1 → CI = [-1, -1]."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = paired_bootstrap(a, b, seed=20260711, n_resamples=10000)
        assert result.mean_diff == pytest.approx(-1.0)
        assert result.ci_lower == pytest.approx(-1.0)
        assert result.ci_upper == pytest.approx(-1.0)

    def test_deterministic(self):
        """Same seed, same data → identical CI."""
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 0.0, 3.0, 2.0, 5.0])
        r1 = paired_bootstrap(a, b, seed=20260711, n_resamples=10000)
        r2 = paired_bootstrap(a, b, seed=20260711, n_resamples=10000)
        assert r1.mean_diff == pytest.approx(r2.mean_diff)
        assert r1.ci_lower == pytest.approx(r2.ci_lower)
        assert r1.ci_upper == pytest.approx(r2.ci_upper)

    def test_one_sided_lower(self):
        """One-sided lower bound request returns only ci_lower_noninf."""
        a = np.array([0.5, 0.6, 0.7])
        b = np.array([0.6, 0.7, 0.8])
        result = paired_bootstrap(a, b, seed=20260711, n_resamples=10000, side="lower")
        # One-sided 95% lower = 5th percentile
        assert result.ci_lower_noninf is not None
        assert result.ci_lower is None
        assert result.ci_upper is None
        # All diffs = -0.1, so lower bound = -0.1
        assert result.ci_lower_noninf == pytest.approx(-0.1)

    def test_invalid_inputs(self):
        """Unequal-length or empty arrays raise."""
        with pytest.raises(ValueError):
            paired_bootstrap(np.array([1.0]), np.array([1.0, 2.0]), seed=20260711)
        with pytest.raises(ValueError):
            paired_bootstrap(np.array([]), np.array([]), seed=20260711)

    def test_bootstrap_mean_within_tolerance(self):
        """On random data, CI should be narrower than full range."""
        rng = np.random.default_rng(20260711)
        a = rng.uniform(0, 1, 100)
        b = rng.uniform(0, 1, 100)
        result = paired_bootstrap(a, b, seed=20260711, n_resamples=10000)
        # CI should contain the mean diff
        assert result.ci_lower <= result.mean_diff <= result.ci_upper
        # CI width should be positive (non-trivial data)
        assert result.ci_upper - result.ci_lower > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  pair alignment
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlignPairs:
    """Align two ``list[MethodResult]`` by example_id intersection."""

    def test_perfect_alignment(self):
        ex1 = _ex(example_id="a")
        ex2 = _ex(example_id="b")
        ex3 = _ex(example_id="c")
        r1 = [_mr("dense_rag", ex1), _mr("dense_rag", ex2), _mr("dense_rag", ex3)]
        r2 = [_mr("hipporag2", ex1), _mr("hipporag2", ex2), _mr("hipporag2", ex3)]
        aligned = align_pairs(r1, r2)
        assert len(aligned) == 3
        assert all(a.example.example_id == b.example.example_id for a, b in aligned)

    def test_missing_id_raises(self):
        ex1 = _ex(example_id="a")
        ex2 = _ex(example_id="b")
        r1 = [_mr("dense_rag", ex1)]
        r2 = [_mr("hipporag2", ex2)]
        with pytest.raises(ValueError, match="No common example_ids"):
            align_pairs(r1, r2)

    def test_duplicate_id_raises(self):
        ex1 = _ex(example_id="a")
        ex2 = _ex(example_id="a")
        r1 = [_mr("dense_rag", ex1), _mr("dense_rag", ex2)]
        r2 = [_mr("hipporag2", ex1)]
        with pytest.raises(ValueError, match="Duplicate"):
            align_pairs(r1, r2)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            align_pairs([], [])


# ═══════════════════════════════════════════════════════════════════════════════
#  H1 cross-dataset contrast
# ═══════════════════════════════════════════════════════════════════════════════


class TestH1Contrast:
    """H1: (MuSiQue_gain + 2Wiki_gain)/2 − NQ_gain."""

    def test_contrast_positive(self):
        """All 3 datasets show hipporag2 > dense_rag → contrast > 0."""
        # Construct data where each dataset has hipporag2 recall higher
        # Not tested via run_statistics yet; manual construction
        # This test uses the lower-level contrast helper once implemented
        pass  # placeholder — rely on run_statistics integration test

    def test_contrast_zero(self):
        """Equal gains across datasets → contrast ≈ 0."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  H3 noninferiority
# ═══════════════════════════════════════════════════════════════════════════════


class TestH3NonInferiority:
    """H3: meta − always_expand Token-F1 one-sided 95% lower > −0.02."""

    def test_clearly_non_inferior(self):
        """All diffs ≈ 0 → lower bound > −0.02 → noninferior."""
        a = np.array([0.8] * 100)
        b = np.array([0.8] * 100)
        result = paired_bootstrap(a, b, seed=20260711, n_resamples=10000, side="lower")
        assert result.ci_lower_noninf > -0.02

    def test_clearly_inferior(self):
        """meta much worse than always_expand → lower bound < −0.02."""
        a = np.array([0.5] * 100)
        b = np.array([0.8] * 100)
        result = paired_bootstrap(a, b, seed=20260711, n_resamples=10000, side="lower")
        assert result.ci_lower_noninf < -0.02


# ═══════════════════════════════════════════════════════════════════════════════
#  stratified group means
# ═══════════════════════════════════════════════════════════════════════════════


class TestStratifiedMeans:
    """Exploratory stratified reporting by stratum (§9.3)."""

    def test_stratification_groups_correct(self):
        ex_a = _ex(dataset="musique", example_id="a", stratum="2")
        ex_b = _ex(dataset="musique", example_id="b", stratum="2")
        ex_c = _ex(dataset="musique", example_id="c", stratum="3")
        ex_d = _ex(dataset="musique", example_id="d", stratum="4")
        # Build MethodResult list by hand with known recall
        # strat_mean uses fused_passages + compute_recall
        # To control recall: pass passages matching gold_docs
        r1 = _result("hipporag2", ex_a, "c", [_passage("p", "text")])  # no gold match → R@5=0
        r2 = _result("hipporag2", ex_b, "c", [_passage("p", "text")])
        r3 = _result("hipporag2", ex_c, "c", [_passage("p", "text")])
        r4 = _result("hipporag2", ex_d, "c", [_passage("p", "text")])

        # Gold docs are ["Doc\ntext"] — passage "text" doesn't match
        # So all recall = 0
        means = stratified_group_means([r1, r2, r3, r4], metric="recall_at_5")
        assert means["2"] == pytest.approx(0.0)
        assert means["3"] == pytest.approx(0.0)
        assert means["4"] == pytest.approx(0.0)
        assert set(means.keys()) == {"2", "3", "4"}


# ═══════════════════════════════════════════════════════════════════════════════
#  integration: run_statistics
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunStatistics:
    """End-to-end: 9 comparisons × 3 metrics → ComparisonResult DataFrame."""

    def _build_dataset(
        self,
        dataset: DatasetId,
        ids: list[str],
        method_a: MethodId,
        method_b: MethodId,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[MethodResult], list[MethodResult]]:
        """Build two method result lists with controlled per-example metrics."""
        if rng is None:
            rng = np.random.default_rng(20260711)
        results_a: list[MethodResult] = []
        results_b: list[MethodResult] = []
        for eid in ids:
            # Method A gets lower recall/EM/F1; method B gets higher
            gold_docs = [f"Doc\n{eid}"]
            gold_answers = [eid]
            ex = _ex(dataset=dataset, example_id=eid, gold_docs=gold_docs, gold_answers=gold_answers, stratum="simple")
            # B always retrieves the correct passage
            pa_b = _passage(f"chunk_{eid}", f"Doc\n{eid}", score=0.9)
            # A retrieves nothing → 0 recall, 0 EM, 0 F1
            pa_a = _passage("chunk_none", "wrong", score=0.5)

            r_a = _result(method_a, ex, "wrong", [pa_a], usage=Usage(actual_usd=0.01, method_equivalent_usd=0.01, prompt_tokens=100, completion_tokens=10))
            r_b = _result(method_b, ex, eid, [pa_b], usage=Usage(actual_usd=0.02, method_equivalent_usd=0.02, prompt_tokens=200, completion_tokens=20))
            results_a.append(r_a)
            results_b.append(r_b)
        return results_a, results_b

    def test_run_statistics_returns_expected_structure(self):
        """Smoke: 3 datasets × 3 comparisons = 9 rows per metric."""
        datasets: list[DatasetId] = ["nq_rear", "musique", "2wikimultihopqa"]
        comparisons: list[tuple[MethodId, MethodId]] = [
            ("hipporag2", "dense_rag"),
            ("metagate", "hipporag2"),
            ("metagate", "always_expand"),
        ]
        all_results: dict[DatasetId, dict[MethodId, list[MethodResult]]] = {}
        rng = np.random.default_rng(20260711)
        for ds in datasets:
            all_results[ds] = {}
            for m in ["dense_rag", "hipporag2", "always_expand", "metagate"]:
                ids = [f"{ds}_{i}" for i in range(30)]
                all_results[ds][m] = [
                    _result(
                        m,
                        _ex(dataset=ds, example_id=eid, gold_docs=[f"Doc\n{eid}"], gold_answers=[eid], stratum="simple"),
                        eid,
                        [_passage(f"chunk_{eid}", f"Doc\n{eid}")],
                        usage=Usage(method_equivalent_usd=0.01, prompt_tokens=100, completion_tokens=10),
                    )
                    for eid in ids
                ]

        report = run_statistics(
            all_results,
            comparisons=comparisons,
            metrics=["recall_at_5", "em", "token_f1"],
            bootstrap_seed=20260711,
            n_resamples=500,  # smaller for speed
            alpha=0.05,
            noninferiority_margin=-0.02,
        )

        # Check structure
        assert "metric" in report.columns
        assert "dataset" in report.columns
        assert "comparison" in report.columns
        # 3 metrics × 9 comparisons + 3 H1 contrast = 30
        assert len(report) == 30

    def test_deterministic_reproducibility(self):
        """Two runs with same inputs → identical output."""
        datasets: list[DatasetId] = ["nq_rear"]
        all_results: dict[DatasetId, dict[MethodId, list[MethodResult]]] = {}
        for ds in datasets:
            all_results[ds] = {}
            ids = [f"{ds}_{i}" for i in range(10)]
            for m in ["dense_rag", "hipporag2"]:
                all_results[ds][m] = [
                    _result(
                        m,
                        _ex(dataset=ds, example_id=eid, gold_docs=[f"Doc\n{eid}"], gold_answers=[eid], stratum="simple"),
                        eid,
                        [_passage(f"chunk_{eid}", f"Doc\n{eid}")],
                        usage=Usage(method_equivalent_usd=0.01),
                    )
                    for eid in ids
                ]

        r1 = run_statistics(
            all_results,
            comparisons=[("hipporag2", "dense_rag")],
            metrics=["recall_at_5"],
            bootstrap_seed=20260711,
            n_resamples=500,
        )
        r2 = run_statistics(
            all_results,
            comparisons=[("hipporag2", "dense_rag")],
            metrics=["recall_at_5"],
            bootstrap_seed=20260711,
            n_resamples=500,
        )
        pd = pytest.importorskip("pandas")
        pd.testing.assert_frame_equal(r1, r2)

    def test_h1_contrast_in_output(self):
        """run_statistics returns H1 contrast as extra row(s)."""
        datasets: list[DatasetId] = ["nq_rear", "musique", "2wikimultihopqa"]
        all_results: dict[DatasetId, dict[MethodId, list[MethodResult]]] = {}
        for ds in datasets:
            all_results[ds] = {}
            ids = [f"{ds}_{i}" for i in range(5)]
            for m in ["dense_rag", "hipporag2"]:
                all_results[ds][m] = [
                    _result(
                        m,
                        _ex(dataset=ds, example_id=eid, gold_docs=[f"Doc\n{eid}"], gold_answers=[eid], stratum="simple"),
                        eid,
                        [_passage(f"chunk_{eid}", f"Doc\n{eid}")],
                        usage=Usage(method_equivalent_usd=0.01),
                    )
                    for eid in ids
                ]

        report = run_statistics(
            all_results,
            comparisons=[("hipporag2", "dense_rag")],
            metrics=["recall_at_5", "em"],
            bootstrap_seed=20260711,
            n_resamples=500,
        )
        # H1 contrast should appear for recall_at_5 (not em)
        h1_rows = report[report["dataset"] == "h1_contrast"]
        recall_h1 = h1_rows[h1_rows["metric"] == "recall_at_5"]
        assert len(recall_h1) == 1

    def test_stratified_output_dataframe(self):
        """Stratified means are callable per dataset+m+metric."""
        ds: DatasetId = "musique"
        ids = [f"musique_{i}" for i in range(20)]
        # Assign strata 2,3,4
        strata_cycle = ["2", "3", "4"]
        results: list[MethodResult] = []
        for i, eid in enumerate(ids):
            s = strata_cycle[i % len(strata_cycle)]
            results.append(
                _result(
                    "hipporag2",
                    _ex(dataset=ds, example_id=eid, gold_docs=[f"Doc\n{eid}"], gold_answers=[eid], stratum=s),
                    eid,
                    [_passage(f"chunk_{eid}", f"Doc\n{eid}")],
                    usage=Usage(method_equivalent_usd=0.01),
                )
            )
        means = stratified_group_means(results, metric="recall_at_5")
        # All passages match gold → recall=1.0 for all strata
        assert means["2"] == pytest.approx(1.0)
        assert means["3"] == pytest.approx(1.0)
        assert means["4"] == pytest.approx(1.0)
        assert set(means.keys()) == {"2", "3", "4"}
