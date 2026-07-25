"""Frozen statistical pipeline — paired bootstrap, exact McNemar, Holm correction.

Implements the pre-registered analysis in design doc §9.3:

- Paired per-question bootstrap (seed 20260711, 10 000 resamples, 95 % percentile CI)
- Two-sided exact McNemar test for per-question EM
- Holm-Bonferroni correction over the 9 primary-comparison p-values (α = 0.05)
- H1 cross-dataset contrast: (MuSiQue_gain + 2Wiki_gain) / 2 − NQ_gain
- H3 one-sided noninferiority lower bound vs −0.02

All results are returned as a ``pandas.DataFrame`` so downstream consumers
(Task 12 CLI / Task 14 tables) can pivot and format without re-running computation.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import binom

from .evaluation import compute_em, compute_f1, compute_recall
from .models import DatasetId, MethodId, MethodResult

# ── Constants (frozen) ────────────────────────────────────────────────────────

_METRICS_PER_EXAMPLE = frozenset(
    ["recall_at_5", "em", "token_f1", "method_equivalent_cost_usd", "llm_calls"]
)
_MERGE_HOW = "inner"


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BootstrapResult:
    """Result of one paired bootstrap run."""

    mean_diff: float
    ci_lower: float | None  # None for one-sided (lower)-only
    ci_upper: float | None  # None for one-sided (lower)-only
    ci_lower_noninf: float | None = None  # one-sided 5th percentile
    n_pairs: int = 0


@dataclass(frozen=True)
class ComparisonResult:
    """One method-pair comparison for one metric/dataset."""

    metric: str
    dataset: str  # dataset id or "h1_contrast"
    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lower: float | None
    ci_upper: float | None
    ci_lower_noninf: float | None  # H3 only
    mcnemar_p: float | None  # EM only
    holm_p: float | None  # EM only (adjusted)
    holm_reject: bool | None  # EM only
    n_pairs: int


# ── Metrics extraction ────────────────────────────────────────────────────────


def _extract_per_example(result: MethodResult) -> dict[str, float]:
    """Extract per-example scalar metrics from one ``MethodResult``.

    Re-uses evaluation module functions to guarantee consistency
    with ``EvaluationReport`` aggregates.
    """
    # Retrieval
    fused = result.fused_passages or []
    recalls = compute_recall(fused, result.example.gold_docs) if fused else {"Recall@5": 0.0}

    # QA
    em = compute_em(result.answer, result.example.gold_answers)
    f1_val = compute_f1(result.answer, result.example.gold_answers)

    # Efficiency
    cost = result.usage.method_equivalent_usd
    llm_calls = 1 + len(result.gate_decisions)  # QA + gate rounds

    return {
        "recall_at_5": float(recalls["Recall@5"]),
        "em": em,
        "token_f1": f1_val,
        "method_equivalent_cost_usd": cost,
        "llm_calls": float(llm_calls),
    }


# ── Pair alignment ────────────────────────────────────────────────────────────


def align_pairs(
    results_a: list[MethodResult],
    results_b: list[MethodResult],
) -> list[tuple[MethodResult, MethodResult]]:
    """Align two result lists by ``example_id`` intersection, preserving order.

    Raises ``ValueError`` on duplicate IDs, missing IDs, or empty intersection.
    """
    if not results_a or not results_b:
        raise ValueError("Both result lists must be non-empty")

    ids_a = [r.example.example_id for r in results_a]
    ids_b = [r.example.example_id for r in results_b]

    # Check duplicates
    if len(ids_a) != len(set(ids_a)):
        raise ValueError("Duplicate example_id in results_a")
    if len(ids_b) != len(set(ids_b)):
        raise ValueError("Duplicate example_id in results_b")

    map_a = {r.example.example_id: r for r in results_a}
    map_b = {r.example.example_id: r for r in results_b}

    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        raise ValueError(
            f"No common example_ids between a ({len(ids_a)}) and b ({len(ids_b)})"
        )

    missing_a = set(ids_b) - set(ids_a)
    if missing_a:
        raise ValueError(f"Missing example_ids in results_a: {sorted(missing_a)}")

    missing_b = set(ids_a) - set(ids_b)
    if missing_b:
        raise ValueError(f"Missing example_ids in results_b: {sorted(missing_b)}")

    # Return in the original order of results_a
    aligned: list[tuple[MethodResult, MethodResult]] = []
    for eid in ids_a:
        if eid in map_b:
            aligned.append((map_a[eid], map_b[eid]))
    return aligned


# ── McNemar ───────────────────────────────────────────────────────────────────


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial distribution.

    ``b``: count of pairs where A wrong → B correct.
    ``c``: count of pairs where A correct → B wrong.

    Under H₀, b and c are equally likely, so each discordant pair has
    probability 0.5 of going to b.  The two-sided p is
    ``2 * sum_{k=0}^{min(b,c)} Binom(k | n=b+c, p=0.5)``.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")

    n = b + c
    if n == 0:
        return 1.0

    k_min = min(b, c)
    # sum PMF for k = 0 … k_min
    p_one_tail = sum(binom.pmf(k, n, 0.5) for k in range(k_min + 1))
    return float(min(2.0 * p_one_tail, 1.0))


# ── Holm correction ───────────────────────────────────────────────────────────


def holm_corrected(
    p_values: list[float], alpha: float = 0.05
) -> list[tuple[float, float, bool]]:
    """Holm-Bonferroni step-down correction.

    Returns a list of ``(raw_p, adjusted_p, reject)`` tuples in the same
    order as the input.  ``adjusted_p`` is the Holm-corrected p-value
    (capped at 1.0) and ``reject`` is ``True`` when the null hypothesis
    is rejected at ``alpha``.
    """
    if not p_values:
        return []

    m = len(p_values)
    indexed = list(enumerate(p_values))
    # Sort by p ascending
    sorted_pairs = sorted(indexed, key=lambda x: x[1])
    ranks = {orig_idx: rank for rank, (orig_idx, _) in enumerate(sorted_pairs, start=1)}

    # Step-down: find first k where p(k) > alpha / (m - k + 1)
    max_reject_rank = 0
    for rank, (_, p) in enumerate(sorted_pairs, start=1):
        threshold = alpha / (m - rank + 1)
        if p <= threshold:
            max_reject_rank = rank
        else:
            break

    # Adjusted p-values (step-up, non-decreasing on sorted order)
    sorted_adj = [0.0] * m
    for rank, (_, p) in enumerate(sorted_pairs, start=1):
        # Holm adjusted: min(1, p * (m - rank + 1))
        # Then enforce non-decreasing from end
        sorted_adj[rank - 1] = min(1.0, p * (m - rank + 1))

    # Enforce non-decreasing: each adjusted p is the maximum of all
    # adjusted p-values up to that rank (running max left-to-right).
    for i in range(1, m):
        sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])

    return [
        (
            p_values[idx],
            sorted_adj[ranks[idx] - 1],
            ranks[idx] <= max_reject_rank,
        )
        for idx in range(m)
    ]


# ── Paired bootstrap ─────────────────────────────────────────────────────────


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    side: Literal["two-sided", "lower"] = "two-sided",
) -> BootstrapResult:
    """Per-question paired bootstrap on ``a`` and ``b`` (same length N).

    Draws ``n_resamples`` bootstrap samples of indices (with replacement)
    and computes the mean paired difference ``d̄ = mean(a - b)`` for each
    resample.  Returns the observed mean difference and percentile CI.

    Parameters
    ----------
    a, b: 1-D arrays of equal length.
    seed: RNG seed (deterministic).
    n_resamples: Number of bootstrap replicates.
    confidence_level: Width of the CI (default 0.95).
    side: ``"two-sided"`` → ``(ci_lower, ci_upper)``; ``"lower"`` → fills only
          ``ci_lower_noninf`` for the one-sided 5 % lower bound.

    Returns
    -------
    BootstrapResult
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Arrays must have equal length: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("Arrays must not be empty")

    n = a.size
    rng = np.random.default_rng(seed)
    diffs = a - b
    obs_mean_diff = float(np.mean(diffs))

    # Bootstrap replicates
    replicates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        replicates[i] = np.mean(diffs[idx])

    alpha = 1.0 - confidence_level

    if side == "two-sided":
        lower = float(np.percentile(replicates, 100.0 * alpha / 2.0))
        upper = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))
        return BootstrapResult(
            mean_diff=obs_mean_diff,
            ci_lower=lower,
            ci_upper=upper,
            n_pairs=n,
        )
    else:  # lower
        lower_b = float(np.percentile(replicates, 100.0 * alpha))
        return BootstrapResult(
            mean_diff=obs_mean_diff,
            ci_lower=None,
            ci_upper=None,
            ci_lower_noninf=lower_b,
            n_pairs=n,
        )


# ── Stratified means ─────────────────────────────────────────────────────────


def stratified_group_means(
    results: list[MethodResult], metric: str
) -> OrderedDict[str, float]:
    """Compute per-stratum mean of *metric* (exploratory reporting §9.3).

    Parameters
    ----------
    results: List of results for one (method, dataset, split) combination.
    metric: ``"recall_at_5"`` / ``"em"`` / ``"token_f1"`` / ``"method_equivalent_cost_usd"``
            / ``"llm_calls"``.

    Returns
    -------
    OrderedDict[str, float]
        Stratum label → mean value, in stratum-sorted order.
    """
    if metric not in _METRICS_PER_EXAMPLE:
        raise ValueError(f"Unknown metric: {metric}")

    by_stratum: dict[str, list[float]] = defaultdict(list)
    for r in results:
        ex = _extract_per_example(r)
        s = r.example.stratum
        by_stratum[s].append(ex[metric])

    out: OrderedDict[str, float] = OrderedDict()
    for s in sorted(by_stratum.keys()):
        vals = by_stratum[s]
        if vals:
            out[s] = float(np.mean(vals))
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_statistics(
    all_results: dict[DatasetId, dict[MethodId, list[MethodResult]]],
    *,
    comparisons: list[tuple[MethodId, MethodId]],
    metrics: list[str],
    bootstrap_seed: int = 20260711,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
    noninferiority_margin: float = -0.02,
) -> pd.DataFrame:
    """Run the full frozen statistical pipeline.

    Parameters
    ----------
    all_results:
        Nested dict: ``dataset → method → list[MethodResult]``.
    comparisons:
        Ordered list of ``(method_a, method_b)`` pairs.  ``a`` is the
        "treatment" and ``b`` is the "baseline" (diff = a − b).
    metrics:
        Which per-example metrics to test (e.g. ``["recall_at_5", "em", "token_f1"]``).
    bootstrap_seed:
        Frozen seed (20260711) for paired bootstrap.
    n_resamples:
        Bootstrap replicates (default 10 000).
    confidence_level:
        CI width (default 0.95).
    alpha:
        Family-wise error rate for Holm (default 0.05).
    noninferiority_margin:
        H3 noninferiority bound for Token-F1 (default −0.02).

    Returns
    -------
    pd.DataFrame
        One row per (metric, dataset, comparison).  Contains:
        ``metric, dataset, comparison, mean_a, mean_b, mean_diff, ci_lower, ci_upper,
        ci_lower_noninf, mcnemar_p, holm_p, holm_reject, n_pairs``.
    """
    rows: list[dict] = []

    # Collect datasets where all required methods exist
    all_methods: set[MethodId] = set(chain.from_iterable(comparisons))
    datasets = [
        ds
        for ds in all_results
        if all(m in all_results[ds] for m in all_methods)
    ]

    if not datasets:
        raise ValueError("No dataset has all required methods")

    # ── Per-comparison calculations ──────────────────────────────────────

    # Temporary storage for McNemar p-values (across all datasets+comparisons
    # for the EM metric, to build the Holm family)
    _mcnemar_entries: list[dict] = []  # {row_idx, raw_p}

    for ds in datasets:
        ds_results = all_results[ds]
        for method_a, method_b in comparisons:
            results_a = ds_results[method_a]
            results_b = ds_results[method_b]

            aligned = align_pairs(results_a, results_b)
            n_pairs = len(aligned)
            ex_a = [_extract_per_example(ra) for ra, _ in aligned]
            ex_b = [_extract_per_example(rb) for _, rb in aligned]

            for metric in metrics:
                vals_a = np.array([ex[metric] for ex in ex_a], dtype=np.float64)
                vals_b = np.array([ex[metric] for ex in ex_b], dtype=np.float64)

                # Bootstrap
                if metric == "token_f1" and method_a == "metagate" and method_b == "always_expand":
                    # H3 noninferiority: one-sided lower
                    boot = paired_bootstrap(
                        vals_a, vals_b,
                        seed=bootstrap_seed,
                        n_resamples=n_resamples,
                        confidence_level=confidence_level,
                        side="lower",
                    )
                else:
                    boot = paired_bootstrap(
                        vals_a, vals_b,
                        seed=bootstrap_seed,
                        n_resamples=n_resamples,
                        confidence_level=confidence_level,
                        side="two-sided",
                    )

                # McNemar (EM only)
                mcnemar_p: float | None = None
                if metric == "em":
                    # b: A wrong, B correct  (em_b > em_a)
                    # c: A correct, B wrong   (em_a > em_b)
                    b_count = int(np.sum(vals_a < vals_b))
                    c_count = int(np.sum(vals_a > vals_b))
                    mcnemar_p = mcnemar_exact(b=b_count, c=c_count)

                row = {
                    "metric": metric,
                    "dataset": ds,
                    "comparison": f"{method_a} vs {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_a": float(np.mean(vals_a)),
                    "mean_b": float(np.mean(vals_b)),
                    "mean_diff": boot.mean_diff,
                    "ci_lower": boot.ci_lower,
                    "ci_upper": boot.ci_upper,
                    "ci_lower_noninf": boot.ci_lower_noninf,
                    "mcnemar_p": mcnemar_p,
                    "holm_p": None,
                    "holm_reject": None,
                    "n_pairs": n_pairs,
                }
                rows.append(row)

                if mcnemar_p is not None:
                    _mcnemar_entries.append((len(rows) - 1, mcnemar_p))

    # ── Holm correction over the 9 primary EM McNemar p-values ──────────

    # Only apply Holm when we have the full 9-entry family
    if _mcnemar_entries:
        mcnemar_ps = [v for _, v in _mcnemar_entries]
        # If exactly 9, apply Holm; if fewer (test data), still apply with m=len
        holm_results = holm_corrected(mcnemar_ps, alpha=alpha)
        for (row_idx, _), (_, adj_p, reject) in zip(_mcnemar_entries, holm_results, strict=False):
            rows[row_idx]["holm_p"] = adj_p
            rows[row_idx]["holm_reject"] = reject

    # ── H1 cross-dataset contrast ───────────────────────────────────────

    # H1 requires the hipporag2 vs dense_rag comparison across all three datasets
    if (
        "hipporag2" in all_methods
        and "dense_rag" in all_methods
        and set(datasets) >= {"nq_rear", "musique", "2wikimultihopqa"}
    ):
        for metric in metrics:
            if metric not in ("recall_at_5", "em", "token_f1"):
                continue
            # Gather per-question diffs for each dataset
            dataset_diffs: dict[str, np.ndarray] = {}
            for ds in ["nq_rear", "musique", "2wikimultihopqa"]:
                if ds not in all_results:
                    continue
                ds_r = all_results[ds]
                if "hipporag2" not in ds_r or "dense_rag" not in ds_r:
                    continue
                aligned = align_pairs(ds_r["hipporag2"], ds_r["dense_rag"])
                ex_a = [_extract_per_example(ra) for ra, _ in aligned]
                ex_b = [_extract_per_example(rb) for _, rb in aligned]
                va = np.array([ex[metric] for ex in ex_a])
                vb = np.array([ex[metric] for ex in ex_b])
                dataset_diffs[ds] = va - vb

            # Contrast: (musique + 2wiki) / 2 − nq_rear
            if set(dataset_diffs) >= {"nq_rear", "musique", "2wikimultihopqa"}:
                multihop = np.concatenate(
                    [dataset_diffs["musique"], dataset_diffs["2wikimultihopqa"]]
                )
                single = dataset_diffs["nq_rear"]
                # Resample each source independently
                rng_contrast = np.random.default_rng(bootstrap_seed)
                n_multihop = len(multihop)
                n_single = len(single)
                contrast_reps = np.empty(n_resamples, dtype=np.float64)
                for i in range(n_resamples):
                    idx_m = rng_contrast.integers(0, n_multihop, size=n_multihop)
                    idx_s = rng_contrast.integers(0, n_single, size=n_single)
                    contrast_reps[i] = np.mean(multihop[idx_m]) - np.mean(single[idx_s])

                obs = float(np.mean(multihop) - np.mean(single))
                alpha_ci = 1.0 - confidence_level
                ci_l = float(np.percentile(contrast_reps, 100.0 * alpha_ci / 2.0))
                ci_u = float(
                    np.percentile(contrast_reps, 100.0 * (1.0 - alpha_ci / 2.0))
                )
                rows.append(
                    {
                        "metric": metric,
                        "dataset": "h1_contrast",
                        "comparison": "hipporag2 vs dense_rag",
                        "method_a": "hipporag2",
                        "method_b": "dense_rag",
                        "mean_a": float("nan"),
                        "mean_b": float("nan"),
                        "mean_diff": obs,
                        "ci_lower": ci_l,
                        "ci_upper": ci_u,
                        "ci_lower_noninf": None,
                        "mcnemar_p": None,
                        "holm_p": None,
                        "holm_reject": None,
                        "n_pairs": n_multihop + n_single,
                    }
                )

    return pd.DataFrame(rows)
