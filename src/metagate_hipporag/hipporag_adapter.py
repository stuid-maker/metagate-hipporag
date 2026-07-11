"""Traceable HippoRAG retrieval bridge with usage snapshots.

Provides ``HippoRAGBridge``, a thin wrapper around the pinned upstream engine
that captures every retrieval decision (facts before/after filtering, dense
fallback, chunk-ID mapping) into a ``RetrievalTrace`` and records usage
deltas via the ``UsageLedger``.

Also provides ``wrap_upstream_inference`` to intercept ``engine.llm_model.infer``
and ``engine.rerank_filter.llm_infer_fn`` so that every LLM call is recorded.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from .models import (
    LedgerEntry,
    RetrievalTrace,
    RetrievedPassage,
    Usage,
)
from .provenance import UsageLedger


class HippoRAGBridge:
    """Narrow, traceable retrieval bridge over the pinned upstream HippoRAG engine.

    The bridge owns no state except *engine* and *top_k*.  All retrieval
    decisions go through ``retrieve_with_trace`` (full graph path) or
    ``dense_with_trace`` (dense-only fallback), both of which return an
    immutable ``RetrievalTrace``.  QA is delegated to ``engine.qa()`` via
    ``answer()``.
    """

    def __init__(self, engine: Any, top_k: int = 5) -> None:
        """Wrap an upstream HippoRAG engine.

        Parameters
        ----------
        engine:
            An initialised upstream ``HippoRAG`` instance whose
            ``prepare_retrieval_objects`` has already been called, or will be
            called on the first retrieval.
        top_k:
            Number of passages to return per trace.  Defaults to 5.
        """
        self.engine = engine
        self.top_k = top_k

    # ── Core retrieval ───────────────────────────────────────────────────

    def retrieve_with_trace(self, query: str) -> RetrievalTrace:
        """Run the full graph retrieval path and return a trace.

        Mirrors the pinned upstream ``HippoRAG.retrieve()`` for a single
        query.  When facts survive the reranker the graph-search path is
        used; otherwise the dense passage path is used as fallback.  Every
        fact, passage, score, and error is captured in the trace.
        """
        started = time.perf_counter()

        # Ensure retrieval objects are ready
        if not self.engine.ready_to_retrieve:
            self.engine.prepare_retrieval_objects()

        # Get query embeddings
        self.engine.get_query_embeddings([query])

        # Fact scoring + reranking
        fact_scores = self.engine.get_fact_scores(query)
        fact_indices, facts, rerank_log = self.engine.rerank_facts(
            query, fact_scores
        )

        if facts:
            # Graph search path
            doc_ids, doc_scores = self.engine.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.engine.global_config.linking_top_k,
                query_fact_scores=fact_scores,
                top_k_facts=facts,
                top_k_fact_indices=fact_indices,
                passage_node_weight=self.engine.global_config.passage_node_weight,
            )
            dense_fallback = False
        else:
            # Dense fallback
            doc_ids, doc_scores = self.engine.dense_passage_retrieval(query)
            dense_fallback = True

        # Map doc_ids to passages
        passages = _build_passages(
            engine=self.engine,
            doc_ids=doc_ids,
            doc_scores=doc_scores,
            top_k=self.top_k,
        )

        elapsed = time.perf_counter() - started

        return RetrievalTrace(
            retrieval_query=query,
            passages=passages,
            facts_before_filter=[
                tuple(value)
                for value in rerank_log.get("facts_before_rerank", [])
            ],
            facts_after_filter=[
                tuple(value)
                for value in rerank_log.get("facts_after_rerank", [])
            ],
            used_dense_fallback=dense_fallback,
            filter_error=rerank_log.get("error"),
            usage=Usage(
                observed_latency_seconds=elapsed,
                method_equivalent_latency_seconds=elapsed,
            ),
        )

    def dense_with_trace(self, query: str) -> RetrievalTrace:
        """Run dense-only retrieval and return a trace.

        Skips the fact-scoring and graph-search paths entirely.  Used by the
        ``dense_rag`` method baseline.
        """
        started = time.perf_counter()

        if not self.engine.ready_to_retrieve:
            self.engine.prepare_retrieval_objects()

        self.engine.get_query_embeddings([query])
        doc_ids, doc_scores = self.engine.dense_passage_retrieval(query)

        passages = _build_passages(
            engine=self.engine,
            doc_ids=doc_ids,
            doc_scores=doc_scores,
            top_k=self.top_k,
        )

        elapsed = time.perf_counter() - started

        return RetrievalTrace(
            retrieval_query=query,
            passages=passages,
            facts_before_filter=[],
            facts_after_filter=[],
            used_dense_fallback=True,
            usage=Usage(
                observed_latency_seconds=elapsed,
                method_equivalent_latency_seconds=elapsed,
            ),
        )

    # ── QA bridge ────────────────────────────────────────────────────────

    def answer(
        self,
        original_question: str,
        passages: list[RetrievedPassage],
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        """Construct a QuerySolution and delegate QA to the upstream engine.

        The upstream ``engine.qa()`` expects a list of ``QuerySolution``
        objects, each carrying ``.question``, ``.docs``, and ``.doc_scores``.
        This method builds one such object from the bridge's own passage
        representation and delegates.

        Returns the upstream ``(query_solutions, response_messages, metadata)``
        tuple.
        """
        # Lazy import — the upstream may not be importable in every env
        from hipporag.utils.misc_utils import QuerySolution  # type: ignore[import-untyped]

        doc_texts = [p.text for p in passages]
        doc_scores_arr = np.array([p.score for p in passages], dtype=np.float64)

        query_solution = QuerySolution(
            question=original_question,
            docs=doc_texts,
            doc_scores=doc_scores_arr,
        )

        return self.engine.qa([query_solution])

    # ── Usage snapshots (ledger-aware) ───────────────────────────────────

    def retrieve_with_ledger(
        self,
        query: str,
        ledger: UsageLedger,
        *,
        dataset: str | None = None,
        example_id: str | None = None,
        method: str | None = None,
        llm_model_name: str = "gpt-4o-mini-2024-07-18",
    ) -> RetrievalTrace:
        """Run ``retrieve_with_trace`` and record LLM usage in the ledger.

        Takes a ``UsageLedger.snapshot()`` before and after the call,
        computing the delta and appending a ``LedgerEntry`` for the
        retrieval-stage LLM usage.
        """
        before = ledger.snapshot()
        trace = self.retrieve_with_trace(query)
        after = ledger.snapshot()
        usage = ledger.delta(before, after)

        # Record the retrieval LLM usage as a separate ledger entry
        if usage.prompt_tokens > 0 or usage.completion_tokens > 0:
            event_id = f"ret-{uuid.uuid4().hex[:16]}"
            entry = LedgerEntry(
                event_id=event_id,
                reservation_id=f"resv-{uuid.uuid4().hex[:12]}",
                stage="retrieval_llm",
                dataset=dataset,  # type: ignore[arg-type]
                example_id=example_id,
                method=method,  # type: ignore[arg-type]
                model=llm_model_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cache_hit=usage.cache_hit,
                batch_discount_applied=False,
                actual_usd=usage.actual_usd,
                method_equivalent_usd=usage.method_equivalent_usd,
                observed_latency_seconds=usage.observed_latency_seconds,
                method_equivalent_latency_seconds=usage.method_equivalent_latency_seconds,
            )
            ledger.append(entry)

        return trace


# ── Upstream inference wrapper ────────────────────────────────────────────────


def wrap_upstream_inference(
    engine: Any,
    on_call: Callable[[str, dict[str, Any], bool], None] | None = None,
) -> Callable[[list[dict[str, str]]], tuple[str, dict[str, Any], bool]]:
    """Install a recording wrapper around ``engine.llm_model.infer``.

    The wrapper records every LLM call's response, metadata, and cache-hit
    status, then returns the exact upstream tuple unchanged.  The same
    wrapper is also installed as ``engine.rerank_filter.llm_infer_fn`` so
    that fact-reranking LLM calls are captured.

    Parameters
    ----------
    engine:
        An upstream ``HippoRAG`` engine whose ``llm_model.infer`` will be
        wrapped.
    on_call:
        Optional callback invoked with ``(response, metadata, cache_hit)``
        after every upstream inference call.  Useful for recording usage.

    Returns
    -------
    wrapper:
        The wrapper function that can be used as a drop-in replacement for
        ``engine.llm_model.infer``.
    """
    original_infer = engine.llm_model.infer

    def _wrapped_infer(
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any], bool]:
        result = original_infer(messages)
        if on_call is not None:
            response, metadata, cache_hit = result
            on_call(response, metadata, cache_hit)
        return result

    # Install on the engine
    engine.llm_model.infer = _wrapped_infer  # type: ignore[method-assign]
    engine.rerank_filter.llm_infer_fn = _wrapped_infer  # type: ignore[assignment]

    return _wrapped_infer


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_passages(
    engine: Any,
    doc_ids: np.ndarray,
    doc_scores: np.ndarray,
    top_k: int,
) -> list[RetrievedPassage]:
    """Map ranked document indices to ``RetrievedPassage`` instances."""
    passages: list[RetrievedPassage] = []
    for rank, (doc_id, score) in enumerate(
        zip(doc_ids[:top_k], doc_scores[:top_k], strict=False), start=1
    ):
        chunk_id = engine.passage_node_keys[int(doc_id)]
        text = engine.chunk_embedding_store.get_row(chunk_id)["content"]
        passages.append(
            RetrievedPassage(
                chunk_id=chunk_id,
                text=text,
                score=float(score),
                rank=rank,
            )
        )
    return passages
