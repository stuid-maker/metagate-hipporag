"""Evaluation metrics: retrieval, QA, calibration, coverage, risk, gate-efficiency.

Provides:
- ``normalize_answer`` — SQuAD-style text normalisation (matches upstream)
- ``compute_recall`` — Recall@k from retrieved passages vs gold docs
- ``compute_em`` / ``compute_f1`` — standard QA metrics
- ``compute_brier`` / ``compute_ece`` — probabilistic calibration metrics
- ``evaluate`` — aggregate pipeline: takes ``list[MethodResult]`` → ``EvaluationReport``

All retrieval matching uses exact string comparison on ``title\\ntext`` format
(config ``retrieval_match=exact_upstream_title_newline_text``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import RetrievedPassage


# ── SQuAD-style normalisation (matches upstream eval_utils.normalize_answer) ──


def normalize_answer(text: str) -> str:
    """Normalize a string: lowercase, remove punctuation, remove articles, collapse whitespace.

    Matches upstream ``hipporag.utils.eval_utils.normalize_answer``.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


# ── Per-example retrieval metrics ─────────────────────────────────────────────


def _retrieval_texts(passages: list[RetrievedPassage]) -> list[str]:
    """Extract deduplicated passage texts in rank order."""
    seen: set[str] = set()
    texts: list[str] = []
    for p in passages:
        if p.text not in seen:
            seen.add(p.text)
            texts.append(p.text)
    return texts


def compute_recall(
    passages: list[RetrievedPassage],
    gold_docs: list[str],
) -> dict[str, float]:
    """Compute Recall@2 and Recall@5 for one example.

    Matches retrieved passage texts against gold document strings using
    exact set intersection (same as upstream ``RetrievalRecall``).
    """
    # Deduplicate gold docs (upstream uses set)
    gold_set: set[str] = set(gold_docs)
    n_gold = len(gold_set)

    if n_gold == 0:
        return {"Recall@2": 0.0, "Recall@5": 0.0}

    retrieved = _retrieval_texts(passages)

    def _recall_at(k: int) -> float:
        top_k = set(retrieved[:k])
        return len(top_k & gold_set) / n_gold

    return {
        "Recall@2": _recall_at(2),
        "Recall@5": _recall_at(5),
    }


# ── Per-example QA metrics ────────────────────────────────────────────────────


def compute_em(predicted: str, gold_answers: list[str]) -> float:
    """Exact match: max over gold answers (normalized)."""
    if not gold_answers:
        raise ValueError("gold_answers must not be empty")
    norm_pred = normalize_answer(predicted)
    scores = [1.0 if normalize_answer(gold) == norm_pred else 0.0 for gold in gold_answers]
    return float(max(scores))


def _token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 between one predicted string and one gold string."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_f1(predicted: str, gold_answers: list[str]) -> float:
    """Token-F1: max over gold answers."""
    if not gold_answers:
        raise ValueError("gold_answers must not be empty")
    scores = [_token_f1(predicted, gold) for gold in gold_answers]
    return float(max(scores))


# ── Calibration metrics ──────────────────────────────────────────────────────


def compute_brier(probs: list[float], labels: list[int]) -> float:
    """Brier score for binary predictions.

    ``probs`` and ``labels`` must have the same non-zero length.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if len(probs) == 0:
        raise ValueError("empty input")
    arr_probs = np.asarray(probs, dtype=np.float64)
    arr_labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((arr_probs - arr_labels) ** 2))


def compute_ece(
    probs: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-frequency binning.

    Sorts predictions by probability, assigns equal numbers of samples to
    each bin, then computes:

        ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    where ``acc`` is the mean label and ``conf`` is the mean predicted
    probability in bin *b*.  Empty bins contribute 0.

    ``n_bins=10`` matches ``statistics.ece_bins=10`` in the frozen config.
    """
    n = len(probs)
    if n != len(labels):
        raise ValueError("probs and labels must have the same length")
    if n == 0:
        raise ValueError("empty input")

    arr_probs = np.asarray(probs, dtype=np.float64)
    arr_labels = np.asarray(labels, dtype=np.float64)

    # Sort by predicted probability
    order = np.argsort(arr_probs)
    sorted_probs = arr_probs[order]
    sorted_labels = arr_labels[order]

    ece = 0.0
    for b in range(n_bins):
        start = int(np.around(b * n / n_bins))
        end = int(np.around((b + 1) * n / n_bins))
        if start >= end:
            continue  # empty bin
        bin_probs = sorted_probs[start:end]
        bin_labels = sorted_labels[start:end]
        bin_size = len(bin_probs)
        acc = float(np.mean(bin_labels))
        conf = float(np.mean(bin_probs))
        ece += (bin_size / n) * abs(acc - conf)

    return ece


# ── Report data structures ────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateEfficiency:
    """Gate decision quality and cost-efficiency metrics.

    Only available for methods that use a gate (``metagate``, ``always_expand``).
    """

    # Decision quality
    stop_correctness: float
    """Proportion of gate-stop decisions where first-round Recall@5 == 1."""

    expand_benefit_rate: float
    """Proportion of expanded examples where fused Recall@5 > first-round Recall@5."""

    delta_recall_expand: float
    """Mean ΔRecall@5 on expanded examples (fused − first)."""

    # Cost
    gate_cost_ratio: float
    """Gate token cost as fraction of total token cost."""

    cost_per_saved_expansion: float | None
    """(gate_total_cost − always_expand_cost) / saved_expansions, or None if missing ref."""

    npv_first_gate: float
    """Negative predictive value: P(sufficient | gate says stop)."""

    # Counts for transparency
    total_samples: int = 0
    stopped_count: int = 0
    expanded_count: int = 0
    stopped_sufficient: int = 0
    expanded_benefited: int = 0


@dataclass(frozen=True)
class EvaluationReport:
    """Complete per-method evaluation report.

    Optional fields (``None``) indicate the metric is not applicable
    for this method (e.g. recall for LLM-only, Brier for non-gate methods).
    """

    # ── metadata ──
    method: str
    dataset: str
    split: str
    n_samples: int
    run_id: str

    # ── retrieval ──
    recall_at_2: float | None
    recall_at_5: float | None
    first_recall_at_2: float | None = None
    first_recall_at_5: float | None = None

    # ── QA ──
    em: float = 0.0
    token_f1: float = 0.0
    forced_em: float = 0.0
    forced_token_f1: float = 0.0

    # ── calibration (first-gate only) ──
    brier: float | None = None
    ece: float | None = None

    # ── meta-cognitive control (metagate only) ──
    expansion_rate: float = 0.0
    false_stop_rate_insufficient: float | None = None
    false_stop_count: int = 0
    first_round_insufficient_count: int = 0
    unnecessary_expansion_rate_sufficient: float | None = None
    unnecessary_expansion_count: int = 0
    first_round_sufficient_count: int = 0

    # ── selective risk (metagate only) ──
    coverage: float = 1.0
    selective_risk_em: float = 0.0
    selective_risk_token_f1: float = 0.0

    # ── gate-efficiency (metagate, always_expand) ──
    gate_efficiency: GateEfficiency | None = None

    # ── efficiency ──
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_embedding_tokens: int = 0
    total_observed_latency_s: float = 0.0
    total_method_equivalent_latency_s: float = 0.0
    cache_hit_count: int = 0
    actual_cost_usd: float = 0.0
    method_equivalent_cost_usd: float = 0.0


# ── Aggregate evaluation ──────────────────────────────────────────────────────


def evaluate(
    results: list[MethodResult],
    *,
    gate_threshold: float,
    ece_bins: int = 10,
    always_expand_results: list[MethodResult] | None = None,
) -> EvaluationReport:
    """Aggregate per-example metrics into an ``EvaluationReport``.

    Parameters
    ----------
    results:
        Ordered list of ``MethodResult`` (one method, one dataset, one split).
    gate_threshold:
        The tuned gate probability threshold for false-stop / expansion analysis.
        Must be in [0, 1].
    ece_bins:
        Number of equal-frequency bins for ECE (default 10).
    always_expand_results:
        Optional always_expand results for the same dataset/split, used to
        compute ``cost_per_saved_expansion``.  When omitted, the gate-efficiency
        cost-per-saved metric is ``None``.

    Returns
    -------
    EvaluationReport
        Aggregated report.
    """
    if not results:
        raise ValueError("results must not be empty")

    method = results[0].method
    dataset = results[0].example.dataset
    split = "dev" if len(results) <= 100 else "test"  # heuristic
    run_id = results[0].run_id

    # ── Per-example collections ──────────────────────────────────────────────

    recall2_vals: list[float] = []
    recall5_vals: list[float] = []
    first_recall2_vals: list[float] = []
    first_recall5_vals: list[float] = []
    em_vals: list[float] = []
    f1_vals: list[float] = []
    gate_probs: list[float] = []
    first_sufficient: list[int] = []  # 1 if first-round R@5==1, else 0
    expanded_flags: list[int] = []  # 1 if expanded, 0 otherwise
    abstain_flags: list[int] = []  # 1 if abstained

    # For false-stop: only metagate
    false_stop_num = 0
    insufficient_count = 0
    unnecessary_num = 0
    sufficient_count = 0

    # For gate-efficiency stop correctness
    stopped_num = 0
    stopped_sufficient_num = 0
    expanded_num = 0
    expanded_benefited_num = 0
    expanded_recall5_deltas: list[float] = []

    # Usage aggregates
    total_prompt = 0
    total_completion = 0
    total_embedding = 0
    total_obs_lat = 0.0
    total_meq_lat = 0.0
    cache_hits = 0
    actual_cost = 0.0
    method_cost = 0.0

    for r in results:
        # ── Fused recall (final retrieval) ───────────────────────────────
        if method == "llm_only":
            # LLM-only: recall is N/A
            pass
        else:
            fused_passages = r.fused_passages if r.fused_passages else []
            fused_recalls = compute_recall(fused_passages, r.example.gold_docs)
            recall2_vals.append(fused_recalls["Recall@2"])
            recall5_vals.append(fused_recalls["Recall@5"])

        # ── First-round recall ───────────────────────────────────────────
        first_passages = (
            r.first_retrieval.passages if r.first_retrieval is not None else []
        )
        if first_passages:
            first_recalls = compute_recall(first_passages, r.example.gold_docs)
            first_recall2_vals.append(first_recalls["Recall@2"])
            first_recall5_vals.append(first_recalls["Recall@5"])
            first_r5 = first_recalls["Recall@5"]
        else:
            first_r5 = 0.0

        # ── QA ───────────────────────────────────────────────────────────
        em_vals.append(compute_em(r.answer, r.example.gold_answers))
        f1_vals.append(compute_f1(r.answer, r.example.gold_answers))

        # ── Gate probability ─────────────────────────────────────────────
        # First gate decision from gate_decisions
        has_gate = len(r.gate_decisions) > 0
        if has_gate:
            prob = r.gate_decisions[0].evidence_sufficient_probability
            gate_probs.append(prob)
            # Label: 1 if first-round Recall@5 == 1
            label = 1 if first_r5 == 1.0 else 0
            first_sufficient.append(label)

        # ── Expansion / abstain ──────────────────────────────────────────
        expanded_flags.append(1 if r.expanded else 0)
        abstain_flags.append(1 if r.abstain_flag else 0)

        # ── False-stop analysis (metagate only) ──────────────────────────
        if method == "metagate" and has_gate:
            prob = r.gate_decisions[0].evidence_sufficient_probability
            r5_sufficient = first_r5 == 1.0

            if r5_sufficient:
                sufficient_count += 1
                # Unnecessary expansion: gate said expand but first was sufficient
                if r.expanded:
                    unnecessary_num += 1
            else:
                insufficient_count += 1
                # False stop: gate said stop but first was insufficient
                if not r.expanded:
                    false_stop_num += 1

            # ── Stop correctness ─────────────────────────────────────────
            gate_said_stop = not r.expanded
            if gate_said_stop:
                stopped_num += 1
                if r5_sufficient:
                    stopped_sufficient_num += 1
            else:
                expanded_num += 1
                # Benefit: fused recall > first recall
                if r.fused_passages:
                    fused_rec = compute_recall(
                        r.fused_passages, r.example.gold_docs
                    )
                    fused_r5 = fused_rec["Recall@5"]
                else:
                    fused_r5 = first_r5
                if fused_r5 > first_r5:
                    expanded_benefited_num += 1
                expanded_recall5_deltas.append(fused_r5 - first_r5)

        # ── Usage ────────────────────────────────────────────────────────
        u = r.usage
        total_prompt += u.prompt_tokens
        total_completion += u.completion_tokens
        total_embedding += u.embedding_tokens
        total_obs_lat += u.observed_latency_seconds
        total_meq_lat += u.method_equivalent_latency_seconds
        if u.cache_hit:
            cache_hits += 1
        actual_cost += u.actual_usd
        method_cost += u.method_equivalent_usd

    n = len(results)

    # ── Aggregate retrieval ──────────────────────────────────────────────────
    if method == "llm_only":
        recall_at_2 = None
        recall_at_5 = None
    else:
        recall_at_2 = float(np.mean(recall2_vals)) if recall2_vals else 0.0
        recall_at_5 = float(np.mean(recall5_vals)) if recall5_vals else 0.0

    first_recall_at_2 = (
        float(np.mean(first_recall2_vals)) if first_recall2_vals else None
    )
    first_recall_at_5 = (
        float(np.mean(first_recall5_vals)) if first_recall5_vals else None
    )

    # ── Aggregate QA ─────────────────────────────────────────────────────────
    em = float(np.mean(em_vals))
    token_f1 = float(np.mean(f1_vals))

    # Forced-answer only (exclude abstained)
    forced_mask = [i for i, a in enumerate(abstain_flags) if a == 0]
    if forced_mask:
        forced_em = float(np.mean([em_vals[i] for i in forced_mask]))
        forced_f1 = float(np.mean([f1_vals[i] for i in forced_mask]))
    else:
        forced_em = 0.0
        forced_f1 = 0.0

    # ── Calibration ──────────────────────────────────────────────────────────
    if gate_probs and first_sufficient:
        brier = compute_brier(gate_probs, first_sufficient)
        ece = compute_ece(gate_probs, first_sufficient, n_bins=ece_bins)
    else:
        brier = None
        ece = None

    # ── False stop / unnecessary expansion ───────────────────────────────────
    expansion_rate = float(np.mean(expanded_flags))

    if method == "metagate":
        false_stop_rate = (
            false_stop_num / insufficient_count if insufficient_count > 0 else 0.0
        )
        unnecessary_rate = (
            unnecessary_num / sufficient_count if sufficient_count > 0 else 0.0
        )
    else:
        false_stop_rate = None
        unnecessary_rate = None

    # ── Selective risk ───────────────────────────────────────────────────────
    # Coverage: proportion of forced-answer examples
    coverage = float(np.sum([1 - a for a in abstain_flags])) / n if n > 0 else 0.0
    # If coverage == 0 (all abstained), selective risk is undefined
    if coverage == 0.0:
        selective_risk_em = float("nan")
        selective_risk_f1 = float("nan")
    else:
        selective_risk_em = 1.0 - forced_em
        selective_risk_f1 = 1.0 - forced_f1

    # ── Gate efficiency ──────────────────────────────────────────────────────
    gate_efficiency: GateEfficiency | None = None
    if method in ("metagate", "always_expand") and gate_probs:
        # Stop correctness (PVV for "stop" decision)
        stop_correctness = (
            stopped_sufficient_num / stopped_num if stopped_num > 0 else 0.0
        )

        # Expand benefit rate
        expand_benefit = (
            expanded_benefited_num / expanded_num if expanded_num > 0 else 0.0
        )

        # Mean delta recall on expanded examples
        delta_recall_expand = (
            float(np.mean(expanded_recall5_deltas))
            if expanded_recall5_deltas
            else 0.0
        )

        # Gate cost (estimated from gate prompt tokens vs total)
        # Gate calls use roughly similar tokens per call; estimate gate token share
        # We use a heuristic: gate prompt + completion for gate decisions
        # Since gate_decisions length varies (1 or 2), estimate as ~300 tokens per gate call
        # A more accurate approach: gate-related Usage is embedded in total usage
        # For simplicity, estimate gate token cost as proportion of gate_calls / total_calls
        # Better: if always_expand provides reference, compute saved expansions
        gate_cost_ratio = 0.0
        if total_prompt + total_completion > 0:
            # Estimate: each gate call ≈ 300 prompt + 50 completion tokens
            total_gate_calls = sum(len(r.gate_decisions) for r in results)
            est_gate_tokens = total_gate_calls * 350
            gate_cost_ratio = min(est_gate_tokens / (total_prompt + total_completion), 1.0)

        # Cost per saved expansion: requires always_expand reference
        cost_per_saved: float | None = None
        if always_expand_results is not None and method == "metagate":
            ae_expanded = sum(1 for r in always_expand_results if r.expanded)
            mg_expanded = expanded_num
            saved = ae_expanded - mg_expanded
            if saved > 0:
                gate_total_cost = actual_cost * gate_cost_ratio
                cost_per_saved = gate_total_cost / saved
            else:
                cost_per_saved = 0.0

        # NPV: P(sufficient | gate says stop) = 1 - false_stop_rate(gate_stop denominator)
        # Gate said stop + first insufficient = false stop
        # NPV = stopped_sufficient / (stopped_sufficient + false_stop_num)
        if method == "metagate":
            npv_denom = stopped_sufficient_num + false_stop_num
            npv = (
                stopped_sufficient_num / npv_denom if npv_denom > 0 else 0.0
            )
        else:
            npv = 0.0  # always_expand doesn't stop

        gate_efficiency = GateEfficiency(
            stop_correctness=stop_correctness,
            expand_benefit_rate=expand_benefit,
            delta_recall_expand=delta_recall_expand,
            gate_cost_ratio=gate_cost_ratio,
            cost_per_saved_expansion=cost_per_saved,
            npv_first_gate=npv,
            total_samples=n,
            stopped_count=stopped_num,
            expanded_count=expanded_num,
            stopped_sufficient=stopped_sufficient_num,
            expanded_benefited=expanded_benefited_num,
        )

    # ── Build report ─────────────────────────────────────────────────────────
    return EvaluationReport(
        method=method,
        dataset=dataset,
        split=split,
        n_samples=n,
        run_id=run_id,
        recall_at_2=recall_at_2,
        recall_at_5=recall_at_5,
        first_recall_at_2=first_recall_at_2,
        first_recall_at_5=first_recall_at_5,
        em=em,
        token_f1=token_f1,
        forced_em=forced_em,
        forced_token_f1=forced_f1,
        brier=brier,
        ece=ece,
        expansion_rate=expansion_rate,
        false_stop_rate_insufficient=false_stop_rate,
        false_stop_count=false_stop_num,
        first_round_insufficient_count=insufficient_count,
        unnecessary_expansion_rate_sufficient=unnecessary_rate,
        unnecessary_expansion_count=unnecessary_num,
        first_round_sufficient_count=sufficient_count,
        coverage=coverage,
        selective_risk_em=selective_risk_em,
        selective_risk_token_f1=selective_risk_f1,
        gate_efficiency=gate_efficiency,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_embedding_tokens=total_embedding,
        total_observed_latency_s=total_obs_lat,
        total_method_equivalent_latency_s=total_meq_lat,
        cache_hit_count=cache_hits,
        actual_cost_usd=actual_cost,
        method_equivalent_cost_usd=method_cost,
    )
