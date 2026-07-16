"""MetaGate: zero-shot evidence-sufficiency gate with threshold selection.

Provides:
- ``select_threshold`` — dev-only threshold tuning using balanced accuracy.
- ``MetaGate`` — the gating controller that implements a bounded
  one-expansion policy with forced answering and abstain flagging.

The gate prompt is frozen in ``configs/gate_prompt.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.metrics import balanced_accuracy_score

from .models import GateDecision, RetrievalTrace, RetrievedPassage

if TYPE_CHECKING:
    from .openai_client import CachedStructuredClient


# ── Gate prompt management ───────────────────────────────────────────────────


def gate_prompt_config() -> Path:
    """Return the absolute path to the frozen gate prompt config file."""
    return Path(__file__).resolve().parents[2] / "configs" / "gate_prompt.json"


def _load_gate_prompt() -> str:
    """Load the gate system prompt from the frozen config file."""
    data = json.loads(gate_prompt_config().read_text(encoding="utf-8"))
    prompt = data["system_prompt"]
    expected_sha = data["sha256"]
    actual_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Gate prompt SHA-256 mismatch: stored={expected_sha}, "
            f"actual={actual_sha}"
        )
    return prompt


DEFAULT_GATE_PROMPT: str = _load_gate_prompt()


# ── MetaGate decision ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetaGateDecision:
    """Result of a single gate call.

    Attributes
    ----------
    gate:
        The structured ``GateDecision`` returned by the LLM.
    expand:
        Whether the policy should trigger an expansion retrieval (``True``
        when ``gate.evidence_sufficient_probability < threshold``).
    expansion_number:
        0 for the first gate, 1 for the second.
    abstain:
        ``True`` when a second-gate probability remains below the threshold
        (forced answer is still produced; this flag records the warning).
    """

    gate: GateDecision
    expand: bool
    expansion_number: int = 0
    abstain: bool = False


# ── Threshold selection (dev-only) ───────────────────────────────────────────


def select_threshold(
    probabilities: list[float],
    sufficient: list[bool],
    candidates: list[float],
) -> float:
    """Choose the best gate threshold from a set of candidates.

    Evaluates each candidate on the pooled dev-set predictions where the
    target is ``Recall@5 == 1`` (encoded as *sufficient* ground-truth).
    The primary objective is ``sklearn.metrics.balanced_accuracy_score``.
    Ties are broken by:

    1. Lower expansion rate (fewer predictions below threshold).
    2. Higher threshold value.

    Parameters
    ----------
    probabilities:
        Gate probabilities for each dev example, in order.
    sufficient:
        Ground-truth labels: ``True`` when the first retrieval already
        achieves Recall@5 == 1.
    candidates:
        Candidate threshold values to evaluate (must be non-empty).

    Returns
    -------
    float
        The selected threshold.

    Raises
    ------
    ValueError
        If *candidates* is empty, or *probabilities* and *sufficient* have
        different lengths.
    """
    if not candidates:
        raise ValueError("threshold_candidates must not be empty")
    if len(probabilities) != len(sufficient):
        raise ValueError(
            "probabilities and sufficient must have the same length, "
            f"got {len(probabilities)} vs {len(sufficient)}"
        )

    best_threshold: float | None = None
    best_ba: float = -1.0
    best_expansion_rate: float = float("inf")

    n = len(probabilities)

    for candidate in sorted(candidates):
        predictions = [p >= candidate for p in probabilities]
        ba = float(balanced_accuracy_score(sufficient, predictions))
        expansion_rate = sum(1 for p in predictions if not p) / n if n > 0 else 0.0

        better = (
            ba > best_ba
            or (
                ba == best_ba
                and (
                    expansion_rate < best_expansion_rate
                    or (
                        expansion_rate == best_expansion_rate
                        and best_threshold is not None
                        and candidate > best_threshold
                    )
                )
            )
        )

        if better:
            best_ba = ba
            best_expansion_rate = expansion_rate
            best_threshold = candidate

    assert best_threshold is not None, "no threshold selected"
    return best_threshold


# ── Gate message builders ────────────────────────────────────────────────────


def _format_passages(passages: list[RetrievedPassage]) -> str:
    """Format passages with stable numeric labels."""
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] {p.text}")
    return "\n".join(lines)


def _format_facts(facts: list[tuple[str, str, str]], label: str) -> str:
    """Format fact triples as a labelled list."""
    if not facts:
        return f"{label}: (none)"
    fact_lines = [f"{label}:"]
    for s, p, o in facts:
        fact_lines.append(f"  ({s}, {p}, {o})")
    return "\n".join(fact_lines)


def build_first_gate_message(trace: RetrievalTrace) -> list[dict[str, str]]:
    """Build the messages for the first gate call.

    The user message contains the original question, facts before/after
    filtering, and the five retrieved passages with stable numeric labels.
    It must NOT include dataset name, gold answer, method name, or whether
    expansion is mandatory.
    """
    parts: list[str] = []

    parts.append(f"Question: {trace.retrieval_query}")
    parts.append("")

    parts.append(_format_facts(trace.facts_before_filter, "Facts before filtering"))
    parts.append(_format_facts(trace.facts_after_filter, "Facts after filtering"))
    parts.append("")

    parts.append("Retrieved passages:")
    parts.append(_format_passages(trace.passages))

    user_content = "\n".join(parts)

    return [
        {"role": "system", "content": DEFAULT_GATE_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_second_gate_message(
    first_trace: RetrievalTrace,
    second_trace: RetrievalTrace,
    fused_passages: list[RetrievedPassage],
) -> list[dict[str, str]]:
    """Build the messages for the second gate call.

    The user message contains the original question, both retrieval queries,
    each round's before/after fact logs, and the five fused passages.
    """
    parts: list[str] = []

    parts.append(f"Original question: {first_trace.retrieval_query}")
    parts.append(f"Expansion query: {second_trace.retrieval_query}")
    parts.append("")

    parts.append("--- Round 1 ---")
    parts.append(_format_facts(first_trace.facts_before_filter, "Facts before filtering"))
    parts.append(_format_facts(first_trace.facts_after_filter, "Facts after filtering"))
    parts.append("")

    parts.append("--- Round 2 ---")
    parts.append(_format_facts(second_trace.facts_before_filter, "Facts before filtering"))
    parts.append(_format_facts(second_trace.facts_after_filter, "Facts after filtering"))
    parts.append("")

    parts.append("Fused passages (after Reciprocal Rank Fusion):")
    parts.append(_format_passages(fused_passages))

    user_content = "\n".join(parts)

    return [
        {"role": "system", "content": DEFAULT_GATE_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── MetaGate controller ──────────────────────────────────────────────────────


class MetaGate:
    """Zero-shot evidence-sufficiency gate with bounded one-expansion policy.

    The policy:

    1. First bridge retrieval → first gate call.
    2. If ``probability >= threshold``: stop, produce forced answer.
    3. Otherwise: retrieve once using ``retrieval_rewrite``, fuse the two
       rankings via RRF, run a second gate, always produce a forced answer,
       and set ``abstain_flag`` when second probability < threshold.

    ``max_expansions`` must be exactly 1 (validated at construction time).
    """

    def __init__(
        self,
        client: CachedStructuredClient,
        threshold: float,
        *,
        max_expansions: int = 1,
        model: str = "gpt-4o-mini-2024-07-18",
        seed: int = 20260711,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        if max_expansions != 1:
            raise ValueError("max_expansions must be 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.client = client
        self.threshold = threshold
        self.max_expansions = max_expansions
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._expansion_count = 0

    # ── Gate calls ───────────────────────────────────────────────────────

    def decide(
        self,
        first_trace: RetrievalTrace,
        *,
        custom_id: str = "",
    ) -> MetaGateDecision:
        """Run the first gate on the initial retrieval trace.

        Parameters
        ----------
        first_trace:
            The retrieval trace from the first (original-query) retrieval.
        custom_id:
            Stable identifier for the gate call (used for caching).

        Returns
        -------
        MetaGateDecision
            Decision with ``expand=True`` when the gate probability falls
            below the threshold.
        """
        messages = build_first_gate_message(first_trace)

        completion = self.client.complete(
            custom_id=custom_id or f"gate-1-{first_trace.retrieval_query[:40]}",
            model=self.model,
            messages=messages,
            response_model=GateDecision,
            max_completion_tokens=self.max_tokens,
            seed=self.seed,
            temperature=self.temperature,
        )

        gate = completion.value
        prob = gate.evidence_sufficient_probability
        should_expand = prob < self.threshold

        if should_expand:
            self._expansion_count += 1

        return MetaGateDecision(
            gate=gate,
            expand=should_expand,
            expansion_number=0,
        )

    def decide_second(
        self,
        first_trace: RetrievalTrace,
        second_trace: RetrievalTrace,
        fused_passages: list[RetrievedPassage],
        first_decision: MetaGateDecision,
        *,
        custom_id: str = "",
    ) -> MetaGateDecision:
        """Run the second gate after expansion.

        Always produces a decision; sets ``abstain=True`` when the second
        probability remains below the threshold.

        Parameters
        ----------
        first_trace:
            The first retrieval trace (original query).
        second_trace:
            The second retrieval trace (expansion query from first gate).
        fused_passages:
            The RRF-fused passage list from both retrievals.
        first_decision:
            The first gate decision (for context).
        custom_id:
            Stable identifier for the gate call.

        Returns
        -------
        MetaGateDecision
            ``expand`` is always ``False`` (bounded by max_expansions=1).
            ``abstain`` is ``True`` when probability < threshold.
        """
        if self._expansion_count >= self.max_expansions:
            # Already at expansion limit — produce forced answer
            # We still call the gate, but don't expand further
            pass

        messages = build_second_gate_message(
            first_trace, second_trace, fused_passages
        )

        completion = self.client.complete(
            custom_id=custom_id or f"gate-2-{first_trace.retrieval_query[:40]}",
            model=self.model,
            messages=messages,
            response_model=GateDecision,
            max_completion_tokens=self.max_tokens,
            seed=self.seed,
            temperature=self.temperature,
        )

        gate = completion.value
        prob = gate.evidence_sufficient_probability
        below_threshold = prob < self.threshold

        # Bounded policy: no further expansion beyond max_expansions
        should_expand = False

        if should_expand:
            self._expansion_count += 1

        return MetaGateDecision(
            gate=gate,
            expand=should_expand,
            expansion_number=1,
            abstain=below_threshold,
        )
