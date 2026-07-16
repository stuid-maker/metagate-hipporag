"""Five method conditions, shared-call accounting, and resumable per-query execution.

Provides:

* ``ResumableRunner`` — JSONL-backed runner that skips already-completed
  examples and validates configuration hashes across restarts.
* Per-method execution functions:
  ``run_llm_only``, ``run_dense_rag``, ``run_hipporag2``,
  ``run_always_expand``, ``run_metagate``.
* ``run_method`` — high-level orchestrator that iterates examples through
  one method, emitting immutable ``MethodResult`` rows to a JSONL file.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .fusion import reciprocal_rank_fusion
from .models import (
    DatasetId,
    Example,
    GateDecision,
    MethodId,
    MethodResult,
    RetrievalTrace,
    RetrievedPassage,
    Usage,
)
from .provenance import append_jsonl, read_jsonl_recover_tail

if TYPE_CHECKING:
    from .hipporag_adapter import HippoRAGBridge
    from .metagate import MetaGate
    from .openai_client import CachedStructuredClient
    from .provenance import UsageLedger


# ── QA prompt for LLM-only method ────────────────────────────────────────────


def _load_qa_prompt() -> str:
    """Load the frozen QA system prompt for the ``llm_only`` method."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "qa_prompt.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    prompt: str = data["system_prompt"]
    expected_sha: str = data["sha256"]
    actual_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"QA prompt SHA-256 mismatch: stored={expected_sha}, "
            f"actual={actual_sha}"
        )
    return prompt


QA_SYSTEM_PROMPT: str = _load_qa_prompt()


class _QAAnswer(BaseModel):
    """Structured answer returned by the LLM for LLM-only QA."""

    answer: str = Field(..., min_length=1)


# ── Run manifest ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunManifest:
    """Immutable metadata that identifies a single method × dataset × split run.

    The manifest is stored alongside the results JSONL and is checked on
    every resume to ensure the configuration has not changed.
    """

    run_id: str
    effective_config_hash: str
    method: MethodId
    dataset: DatasetId
    split: str
    created_at: str
    gate_threshold: float | None = None


def _compute_run_id(
    effective_config_hash: str,
    method: MethodId,
    dataset: DatasetId,
    split: str,
) -> str:
    """Deterministic short run identifier."""
    payload = f"{effective_config_hash}|{method}|{dataset}|{split}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── ResumableRunner ──────────────────────────────────────────────────────────


class ResumableRunner:
    """JSONL-backed resumable executor for one (method, dataset, split) run.

    On construction the runner reads any existing ``results.jsonl``,
    validates the embedded manifest against *effective_config_hash*, and
    builds a set of already-completed ``example_id`` values.  Subsequent
    ``append()`` calls atomically flush one ``MethodResult`` at a time.

    When the config hash does not match (e.g. a prompt was changed) the old
    run directory is renamed with a ``.stale.`` prefix and a fresh run
    begins.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        effective_config_hash: str,
        method: MethodId,
        dataset: DatasetId,
        split: str,
        *,
        gate_threshold: float | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._results_path = output_dir / "results.jsonl"
        self._manifest_path = output_dir / "run_manifest.json"
        self._run_id = run_id
        self._effective_config_hash = effective_config_hash
        self._method = method
        self._dataset = dataset
        self._split = split
        self._gate_threshold = gate_threshold

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Build or validate the run manifest
        existing = self._read_manifest()
        if existing is not None:
            # Config hash must match
            if existing.effective_config_hash != effective_config_hash:
                self._stale_and_reset()
                self._write_manifest()
                self._completed: set[str] = set()
            else:
                self._completed = self._load_completed_ids()
        else:
            self._write_manifest()
            self._completed = self._load_completed_ids()

    # ── Manifest I/O ─────────────────────────────────────────────────────

    def _read_manifest(self) -> RunManifest | None:
        """Read the run manifest if it exists."""
        if not self._manifest_path.exists():
            return None
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return RunManifest(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _write_manifest(self) -> None:
        """Write a fresh run manifest."""
        manifest = RunManifest(
            run_id=self._run_id,
            effective_config_hash=self._effective_config_hash,
            method=self._method,
            dataset=self._dataset,
            split=self._split,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            gate_threshold=self._gate_threshold,
        )
        self._manifest_path.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _stale_and_reset(self) -> None:
        """Rename the existing directory and start fresh."""
        stale_suffix = time.strftime(".stale.%Y%m%dT%H%M%S", time.gmtime())
        stale_path = self._output_dir.with_name(
            self._output_dir.name + stale_suffix
        )
        self._output_dir.rename(stale_path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Completed-ID tracking ────────────────────────────────────────────

    def _load_completed_ids(self) -> set[str]:
        """Parse existing JSONL rows and return completed example IDs."""
        rows = read_jsonl_recover_tail(self._results_path)
        completed: set[str] = set()
        for row in rows:
            example_id = row.get("example", {}).get("example_id")
            if example_id:
                completed.add(example_id)
        return completed

    @property
    def completed_ids(self) -> frozenset[str]:
        """Frozen set of example IDs already present in the JSONL file."""
        return frozenset(self._completed)

    @property
    def completed_count(self) -> int:
        """Number of already-completed examples."""
        return len(self._completed)

    # ── Append ───────────────────────────────────────────────────────────

    def append(self, result: MethodResult) -> None:
        """Atomically append one ``MethodResult`` to the JSONL file."""
        row = result.model_dump(mode="json")
        append_jsonl(self._results_path, [row])
        self._completed.add(result.example.example_id)


# ── Per-method execution ─────────────────────────────────────────────────────


def _qa_via_upstream(
    question: str,
    passages: list[RetrievedPassage],
    bridge: HippoRAGBridge,
) -> tuple[str, list[str], list[dict]]:
    """Delegate QA to the upstream HippoRAG engine via the bridge."""
    query_solutions, response_messages, metadata = bridge.answer(
        original_question=question, passages=passages
    )
    answer: str = query_solutions[0].answer
    return answer, response_messages, metadata


def _qa_llm_only(
    question: str,
    client: CachedStructuredClient,
    *,
    model: str = "gpt-4o-mini-2024-07-18",
    seed: int = 20260711,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Answer a question using only parametric knowledge (no retrieval)."""
    custom_id = f"qa-llm-only-{uuid.uuid4().hex[:12]}"
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    completion = client.complete(
        custom_id=custom_id,
        model=model,
        messages=messages,
        response_model=_QAAnswer,
        max_completion_tokens=max_tokens,
        seed=seed,
        temperature=temperature,
    )
    return completion.value.answer


# ── Method implementations ───────────────────────────────────────────────────


def run_llm_only(
    example: Example,
    client: CachedStructuredClient,
    ledger: UsageLedger,
    run_id: str,
    *,
    llm_model: str = "gpt-4o-mini-2024-07-18",
    seed: int = 20260711,
    temperature: float = 0.0,
    max_tokens: int = 256,
    **__: Any,
) -> MethodResult:
    """LLM-only baseline: answer from parametric knowledge, no retrieval."""
    before = ledger.snapshot()
    errors: list[str] = []

    try:
        answer = _qa_llm_only(
            question=example.question,
            client=client,
            model=llm_model,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        answer = ""
        errors.append(f"qa-error: {exc}")

    after = ledger.snapshot()
    usage = ledger.delta(before, after)

    return MethodResult(
        run_id=run_id,
        method="llm_only",
        example=example,
        first_retrieval=None,
        second_retrieval=None,
        fused_passages=[],
        answer=answer,
        gate_decisions=[],
        expanded=False,
        abstain_flag=False,
        usage=usage,
        errors=errors,
    )


def run_dense_rag(
    example: Example,
    bridge: HippoRAGBridge,
    client: CachedStructuredClient,
    ledger: UsageLedger,
    run_id: str,
    *,
    top_k: int = 5,
    **__: Any,
) -> MethodResult:
    """Dense RAG baseline: dense-only retrieval → upstream QA."""
    before = ledger.snapshot()
    errors: list[str] = []

    # Retrieve
    try:
        trace = bridge.dense_with_trace(example.question)
    except Exception as exc:
        errors.append(f"retrieval-error: {exc}")
        trace = RetrievalTrace(
            retrieval_query=example.question,
            passages=[],
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
        )

    # QA
    try:
        answer, _, _ = _qa_via_upstream(
            question=example.question,
            passages=trace.passages,
            bridge=bridge,
        )
    except Exception as exc:
        answer = ""
        errors.append(f"qa-error: {exc}")

    after = ledger.snapshot()
    usage = ledger.delta(before, after)

    return MethodResult(
        run_id=run_id,
        method="dense_rag",
        example=example,
        first_retrieval=trace,
        second_retrieval=None,
        fused_passages=trace.passages,
        answer=answer,
        gate_decisions=[],
        expanded=False,
        abstain_flag=False,
        usage=usage,
        errors=errors,
    )


def run_hipporag2(
    example: Example,
    bridge: HippoRAGBridge,
    client: CachedStructuredClient,
    ledger: UsageLedger,
    run_id: str,
    *,
    top_k: int = 5,
    **__: Any,
) -> MethodResult:
    """HippoRAG 2 baseline: full graph retrieval → upstream QA."""
    before = ledger.snapshot()
    errors: list[str] = []

    # Retrieve
    try:
        trace = bridge.retrieve_with_trace(example.question)
    except Exception as exc:
        errors.append(f"retrieval-error: {exc}")
        trace = RetrievalTrace(
            retrieval_query=example.question,
            passages=[],
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
        )

    # QA
    try:
        answer, _, _ = _qa_via_upstream(
            question=example.question,
            passages=trace.passages,
            bridge=bridge,
        )
    except Exception as exc:
        answer = ""
        errors.append(f"qa-error: {exc}")

    after = ledger.snapshot()
    usage = ledger.delta(before, after)

    return MethodResult(
        run_id=run_id,
        method="hipporag2",
        example=example,
        first_retrieval=trace,
        second_retrieval=None,
        fused_passages=trace.passages,
        answer=answer,
        gate_decisions=[],
        expanded=False,
        abstain_flag=False,
        usage=usage,
        errors=errors,
    )


def run_always_expand(
    example: Example,
    bridge: HippoRAGBridge,
    metagate: MetaGate,
    client: CachedStructuredClient,
    ledger: UsageLedger,
    run_id: str,
    *,
    rrf_k: int = 60,
    top_k: int = 5,
    **__: Any,
) -> MethodResult:
    """Always-Expand baseline: first retrieval → always second retrieval → RRF → QA.

    The gate is still called (for logging) but its decision is ignored —
    expansion always happens regardless of confidence.
    """
    before = ledger.snapshot()
    errors: list[str] = []
    gate_decisions: list[GateDecision] = []

    # ── First retrieval ──────────────────────────────────────────────────
    try:
        first_trace = bridge.retrieve_with_trace(example.question)
    except Exception as exc:
        errors.append(f"retrieval-1-error: {exc}")
        first_trace = RetrievalTrace(
            retrieval_query=example.question,
            passages=[],
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
        )

    # ── First gate (always called; expansion decision ignored) ───────────
    try:
        first_decision = metagate.decide(
            first_trace,
            custom_id=f"gate-1-{example.dataset}-{example.example_id}",
        )
        gate_decisions.append(first_decision.gate)
        rewrite_query = first_decision.gate.retrieval_rewrite
    except Exception as exc:
        errors.append(f"gate-1-error: {exc}")
        rewrite_query = example.question  # fallback: re-retrieve with original

    # ── Second retrieval (always) ────────────────────────────────────────
    try:
        second_trace = bridge.retrieve_with_trace(rewrite_query)
    except Exception as exc:
        errors.append(f"retrieval-2-error: {exc}")
        second_trace = RetrievalTrace(
            retrieval_query=rewrite_query,
            passages=[],
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
        )

    # ── Fuse ─────────────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(
        [first_trace.passages, second_trace.passages],
        k=rrf_k,
        top_k=top_k,
    )

    # ── Second gate (always called; only records confidence) ─────────────
    try:
        second_decision = metagate.decide_second(
            first_trace=first_trace,
            second_trace=second_trace,
            fused_passages=fused,
            first_decision=first_decision,
            custom_id=f"gate-2-{example.dataset}-{example.example_id}",
        )
        gate_decisions.append(second_decision.gate)
    except Exception as exc:
        errors.append(f"gate-2-error: {exc}")

    # ── QA ───────────────────────────────────────────────────────────────
    try:
        answer, _, _ = _qa_via_upstream(
            question=example.question,
            passages=fused,
            bridge=bridge,
        )
    except Exception as exc:
        answer = ""
        errors.append(f"qa-error: {exc}")

    after = ledger.snapshot()
    usage = ledger.delta(before, after)

    return MethodResult(
        run_id=run_id,
        method="always_expand",
        example=example,
        first_retrieval=first_trace,
        second_retrieval=second_trace,
        fused_passages=fused,
        answer=answer,
        gate_decisions=gate_decisions,
        expanded=True,
        abstain_flag=False,  # Always-Expand never abstains
        usage=usage,
        errors=errors,
    )


def run_metagate(
    example: Example,
    bridge: HippoRAGBridge,
    metagate: MetaGate,
    client: CachedStructuredClient,
    ledger: UsageLedger,
    run_id: str,
    *,
    rrf_k: int = 60,
    top_k: int = 5,
    **__: Any,
) -> MethodResult:
    """MetaGate method: gate-controlled bounded one-expansion policy.

    1. First retrieval → first gate.
    2. If confident (probability ≥ threshold): stop, produce answer.
    3. Otherwise: second retrieval using rewrite, RRF fusion, second gate,
       always produce answer, set ``abstain_flag`` if second gate is
       still below threshold.
    """
    before = ledger.snapshot()
    errors: list[str] = []
    gate_decisions: list[GateDecision] = []

    # ── First retrieval ──────────────────────────────────────────────────
    try:
        first_trace = bridge.retrieve_with_trace(example.question)
    except Exception as exc:
        errors.append(f"retrieval-1-error: {exc}")
        first_trace = RetrievalTrace(
            retrieval_query=example.question,
            passages=[],
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
        )

    # ── First gate ───────────────────────────────────────────────────────
    try:
        first_decision = metagate.decide(
            first_trace,
            custom_id=f"gate-1-{example.dataset}-{example.example_id}",
        )
        gate_decisions.append(first_decision.gate)
    except Exception as exc:
        errors.append(f"gate-1-error: {exc}")
        # On gate error, treat as confident (don't expand) with empty answer
        after_err = ledger.snapshot()
        usage_err = ledger.delta(before, after_err)
        return MethodResult(
            run_id=run_id,
            method="metagate",
            example=example,
            first_retrieval=first_trace,
            second_retrieval=None,
            fused_passages=first_trace.passages,
            answer="",
            gate_decisions=[],
            expanded=False,
            abstain_flag=False,
            usage=usage_err,
            errors=errors,
        )

    # ── Decision: expand or stop? ────────────────────────────────────────
    if not first_decision.expand:
        # Confident — stop with first-retrieval passages
        try:
            answer, _, _ = _qa_via_upstream(
                question=example.question,
                passages=first_trace.passages,
                bridge=bridge,
            )
        except Exception as exc:
            answer = ""
            errors.append(f"qa-error: {exc}")

        after = ledger.snapshot()
        usage = ledger.delta(before, after)

        return MethodResult(
            run_id=run_id,
            method="metagate",
            example=example,
            first_retrieval=first_trace,
            second_retrieval=None,
            fused_passages=first_trace.passages,
            answer=answer,
            gate_decisions=gate_decisions,
            expanded=False,
            abstain_flag=False,
            usage=usage,
            errors=errors,
        )

    # ── Expand: second retrieval ─────────────────────────────────────────
    rewrite_query = first_decision.gate.retrieval_rewrite
    try:
        second_trace = bridge.retrieve_with_trace(rewrite_query)
    except Exception as exc:
        errors.append(f"retrieval-2-error: {exc}")
        second_trace = RetrievalTrace(
            retrieval_query=rewrite_query,
            passages=[],
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
        )

    # ── Fuse ─────────────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(
        [first_trace.passages, second_trace.passages],
        k=rrf_k,
        top_k=top_k,
    )

    # ── Second gate ──────────────────────────────────────────────────────
    abstain = False
    try:
        second_decision = metagate.decide_second(
            first_trace=first_trace,
            second_trace=second_trace,
            fused_passages=fused,
            first_decision=first_decision,
            custom_id=f"gate-2-{example.dataset}-{example.example_id}",
        )
        gate_decisions.append(second_decision.gate)
        abstain = second_decision.abstain
    except Exception as exc:
        errors.append(f"gate-2-error: {exc}")

    # ── QA (forced — always produce answer) ──────────────────────────────
    try:
        answer, _, _ = _qa_via_upstream(
            question=example.question,
            passages=fused,
            bridge=bridge,
        )
    except Exception as exc:
        answer = ""
        errors.append(f"qa-error: {exc}")

    after = ledger.snapshot()
    usage = ledger.delta(before, after)

    return MethodResult(
        run_id=run_id,
        method="metagate",
        example=example,
        first_retrieval=first_trace,
        second_retrieval=second_trace,
        fused_passages=fused,
        answer=answer,
        gate_decisions=gate_decisions,
        expanded=True,
        abstain_flag=abstain,
        usage=usage,
        errors=errors,
    )


# ── High-level orchestrator ──────────────────────────────────────────────────


# Map method IDs to their execution functions
_METHOD_RUNNERS = {
    "llm_only": run_llm_only,
    "dense_rag": run_dense_rag,
    "hipporag2": run_hipporag2,
    "always_expand": run_always_expand,
    "metagate": run_metagate,
}


def run_method(
    method: MethodId,
    examples: list[Example],
    *,
    bridge: HippoRAGBridge | None = None,
    metagate: MetaGate | None = None,
    client: CachedStructuredClient,
    ledger: UsageLedger,
    output_dir: Path,
    effective_config_hash: str,
    gate_threshold: float | None = None,
    rrf_k: int = 60,
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini-2024-07-18",
    seed: int = 20260711,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> list[MethodResult]:
    """Execute *method* on *examples* with resumable JSONL output.

    Parameters
    ----------
    method:
        One of the five ``MethodId`` values.
    examples:
        Ordered list of ``Example`` instances to run.
    bridge:
        Initialised ``HippoRAGBridge`` (required except for ``llm_only``).
    metagate:
        ``MetaGate`` controller (required for ``always_expand`` and
        ``metagate``).
    client:
        ``CachedStructuredClient`` for gate calls and LLM-only QA.
    ledger:
        ``UsageLedger`` for cost accounting.
    output_dir:
        Directory where ``results.jsonl`` and ``run_manifest.json`` are
        written.
    effective_config_hash:
        Full SHA-256 of the effective experiment configuration.
    gate_threshold:
        The tuned gate threshold (recorded in the run manifest).
    rrf_k:
        RRF smoothing constant (default 60).
    top_k:
        Number of top passages per retrieval and after fusion.
    llm_model, seed, temperature, max_tokens:
        QA parameters forwarded to the LLM client.

    Returns
    -------
    list[MethodResult]
        All results in example order.
    """
    if method not in _METHOD_RUNNERS:
        raise ValueError(f"unknown method: {method!r}")

    # Validate required dependencies per method
    _needs_bridge = {"dense_rag", "hipporag2", "always_expand", "metagate"}
    _needs_metagate = {"always_expand", "metagate"}

    if method in _needs_bridge and bridge is None:
        raise ValueError(f"method {method!r} requires a bridge")
    if method in _needs_metagate and metagate is None:
        raise ValueError(f"method {method!r} requires a MetaGate instance")

    # Determine dataset and split
    if not examples:
        return []
    dataset = examples[0].dataset
    split = "dev" if len(examples) <= 100 else "test"
    run_id = _compute_run_id(effective_config_hash, method, dataset, split)

    # Resumable runner
    runner = ResumableRunner(
        output_dir=output_dir,
        run_id=run_id,
        effective_config_hash=effective_config_hash,
        method=method,
        dataset=dataset,
        split=split,
        gate_threshold=gate_threshold,
    )

    # Find pending examples
    completed = runner.completed_ids
    pending = [ex for ex in examples if ex.example_id not in completed]

    # Build kwargs dict for the method function
    kwargs: dict = {
        "bridge": bridge,
        "metagate": metagate,
        "client": client,
        "ledger": ledger,
        "run_id": run_id,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "llm_model": llm_model,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    runner_fn = _METHOD_RUNNERS[method]

    for example in pending:
        try:
            result = runner_fn(example, **kwargs)
        except Exception as exc:
            after = ledger.snapshot()
            before_dummy = ledger.snapshot()
            try:
                usage_err = ledger.delta(before_dummy, after)
            except Exception:
                usage_err = Usage()
            result = MethodResult(
                run_id=run_id,
                method=method,
                example=example,
                first_retrieval=None,
                second_retrieval=None,
                fused_passages=[],
                answer="",
                gate_decisions=[],
                expanded=False,
                abstain_flag=False,
                usage=usage_err,
                errors=[f"fatal: {exc}"],
            )
        runner.append(result)

    # Collect all results (completed + new) in example order
    all_results: list[MethodResult] = []
    existing_rows = read_jsonl_recover_tail(runner._results_path)
    for row in existing_rows:
        try:
            all_results.append(MethodResult.model_validate(row))
        except Exception:
            continue

    return all_results
