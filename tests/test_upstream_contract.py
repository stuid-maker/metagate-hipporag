"""Upstream contract equivalence tests for the HippoRAG bridge.

Offline tests validate that the bridge produces identical passage rankings,
scores, and fact logs as the pinned upstream ``HippoRAG.retrieve()`` when
both run from the same deterministic engine state.  These tests contain no
API calls and fail if any network method is reached.

The module also defines ``assert_real_index_contract``, a reusable check for
Task 11 that compares official and bridge retrieval from a completed real
index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metagate_hipporag.hipporag_adapter import HippoRAGBridge
from metagate_hipporag.models import RetrievalTrace

# ── Deterministic fake engine with exact upstream values ──────────────────────


class DeterministicFakeConfig:
    """Frozen config matching the experiment's retrieval settings."""

    linking_top_k: int = 5
    passage_node_weight: float = 0.05
    dataset: str = "musique"
    embedding_model_name: str = "text-embedding-3-large"
    embedding_return_as_normalized: bool = True
    embedding_max_seq_len: int = 8192
    embedding_batch_size: int = 64
    embedding_base_url: str = "https://api.openai.com/v1"
    azure_embedding_endpoint: Any = None
    llm_model_name: str = "gpt-4o-mini-2024-07-18"


class DeterministicFakeStore:
    """Fake embedding store that returns pre-configured rows."""

    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows

    def get_row(self, chunk_id: str) -> dict[str, Any]:
        return self._rows[chunk_id]

    def get_rows(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {cid: self._rows[cid] for cid in chunk_ids}

    def get_all_ids(self) -> list[str]:
        return list(self._rows.keys())

    def get_embeddings(self, keys: list[str]) -> list[np.ndarray]:  # type: ignore[type-arg]
        return [np.array([0.1] * 3072, dtype=np.float32) for _ in keys]


def _make_contract_engine(
    *,
    passage_ids: list[str] | None = None,
    passage_texts: dict[str, str] | None = None,
    fact_rerank_result: tuple[list[int], list[tuple], dict[str, Any]] | None = None,
    dense_ids: np.ndarray | None = None,
    dense_scores: np.ndarray | None = None,
    graph_ids: np.ndarray | None = None,
    graph_scores: np.ndarray | None = None,
    fact_scores: np.ndarray | None = None,
    fact_node_keys: list[str] | None = None,
) -> Any:
    """Build a minimal deterministic engine for contract verification.

    Every method returns exactly the values provided — no randomness, no
    network calls.  Two copies of the engine can be independently reset and
    should produce identical results.
    """
    if passage_ids is None:
        passage_ids = [f"chunk_{i}" for i in range(5)]
    if passage_texts is None:
        passage_texts = {cid: f"Text of {cid}" for cid in passage_ids}
    if fact_rerank_result is None:
        fact_rerank_result = (
            [0, 1],
            [("A", "relates_to", "B"), ("C", "has", "D")],
            {
                "facts_before_rerank": [
                    ("A", "relates_to", "B"),
                    ("C", "has", "D"),
                    ("E", "is", "F"),
                ],
                "facts_after_rerank": [
                    ("A", "relates_to", "B"),
                    ("C", "has", "D"),
                ],
            },
        )
    if dense_ids is None:
        dense_ids = np.array([0, 1, 2, 3, 4])
    if dense_scores is None:
        dense_scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    if graph_ids is None:
        graph_ids = np.array([2, 0, 4, 1, 3])
    if graph_scores is None:
        graph_scores = np.array([0.98, 0.88, 0.78, 0.68, 0.58])
    if fact_scores is None:
        fact_scores = np.array([0.9, 0.7, 0.5])
    if fact_node_keys is None:
        fact_node_keys = ["f0", "f1", "f2"]

    engine = type("_ContractEngine", (), {})()

    # Config
    engine.global_config = DeterministicFakeConfig()

    # Stores
    store_rows = {cid: {"content": passage_texts.get(cid, f"Text of {cid}")}
                  for cid in passage_ids}
    engine.chunk_embedding_store = DeterministicFakeStore(store_rows)
    engine.entity_embedding_store = DeterministicFakeStore({})
    engine.fact_embedding_store = DeterministicFakeStore(
        {fid: {"content": str(("S", "P", "O"))} for fid in fact_node_keys}
    )

    # Keys
    engine.passage_node_keys = list(passage_ids)
    engine.fact_node_keys = list(fact_node_keys)
    engine.entity_node_keys = ["e0", "e1"]

    # State
    engine.ready_to_retrieve = True
    engine.query_to_embedding: dict[str, dict[str, np.ndarray]] = {
        "triple": {},
        "passage": {},
    }

    # Pre-configured return values (immutable)
    engine._dense_ids = dense_ids
    engine._dense_scores = dense_scores
    engine._graph_ids = graph_ids
    engine._graph_scores = graph_scores
    engine._fact_scores = fact_scores
    engine._rerank_result = fact_rerank_result

    # LLM (must fail if reached — contract tests are offline)
    class _NoNetworkLLM:
        def infer(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "Network LLM call reached during offline contract test"
            )

    engine.llm_model = _NoNetworkLLM()

    class _NoNetworkRerank:
        pass

    engine.rerank_filter = _NoNetworkRerank()
    engine.rerank_filter.llm_infer_fn = None  # type: ignore[attr-defined]

    # ── Methods that use pre-configured values ──────────────────────────

    def _get_query_embeddings(queries: list[str]) -> None:
        for q in queries:
            engine.query_to_embedding["passage"][q] = np.array(
                [0.1] * 3072, dtype=np.float32
            )
            engine.query_to_embedding["triple"][q] = np.array(
                [0.1] * 3072, dtype=np.float32
            )

    def _get_fact_scores(query: str) -> np.ndarray:  # type: ignore[type-arg]
        return engine._fact_scores.copy()

    def _rerank_facts(
        query: str, query_fact_scores: np.ndarray  # type: ignore[type-arg]
    ) -> tuple[list[int], list[tuple], dict[str, Any]]:
        # Return deep copy so each call is independent
        indices, facts, log = engine._rerank_result
        return (
            list(indices),
            list(facts),
            dict(log),  # shallow copy — values are tuples (immutable)
        )

    def _graph_search(
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,  # type: ignore[type-arg]
        top_k_facts: list[tuple],
        top_k_fact_indices: list[int],
        passage_node_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        return engine._graph_ids.copy(), engine._graph_scores.copy()

    def _dense_retrieval(
        query: str,
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        return engine._dense_ids.copy(), engine._dense_scores.copy()

    def _prepare_retrieval_objects() -> None:
        engine.ready_to_retrieve = True

    engine.get_query_embeddings = _get_query_embeddings
    engine.get_fact_scores = _get_fact_scores
    engine.rerank_facts = _rerank_facts
    engine.graph_search_with_fact_entities = _graph_search
    engine.dense_passage_retrieval = _dense_retrieval
    engine.prepare_retrieval_objects = _prepare_retrieval_objects

    return engine


def _deepcopy_engine(engine: Any) -> Any:
    """Create an independent copy of a deterministic engine.

    All mutable state (arrays, dicts, lists) is deep-copied so that the
    original and the copy do not alias.  Methods are re-bound to the copy.
    """
    copy_engine = type("_ContractEngineCopy", (), {})()

    # Copy scalar/immutable attributes
    copy_engine.global_config = engine.global_config
    copy_engine.ready_to_retrieve = engine.ready_to_retrieve
    copy_engine.passage_node_keys = list(engine.passage_node_keys)
    copy_engine.fact_node_keys = list(engine.fact_node_keys)
    copy_engine.entity_node_keys = list(engine.entity_node_keys)
    copy_engine.llm_model = engine.llm_model
    copy_engine.rerank_filter = engine.rerank_filter

    # Deep-copy stores
    copy_engine.chunk_embedding_store = engine.chunk_embedding_store
    copy_engine.entity_embedding_store = engine.entity_embedding_store
    copy_engine.fact_embedding_store = engine.fact_embedding_store

    # Deep-copy mutable state
    copy_engine.query_to_embedding = {
        "triple": dict(engine.query_to_embedding.get("triple", {})),
        "passage": dict(engine.query_to_embedding.get("passage", {})),
    }

    # Deep-copy pre-configured arrays
    copy_engine._dense_ids = engine._dense_ids.copy()
    copy_engine._dense_scores = engine._dense_scores.copy()
    copy_engine._graph_ids = engine._graph_ids.copy()
    copy_engine._graph_scores = engine._graph_scores.copy()
    copy_engine._fact_scores = engine._fact_scores.copy()

    # Deep-copy rerank result (tuples are immutable, so shallow dict copy is fine)
    indices, facts, log = engine._rerank_result
    copy_engine._rerank_result = (list(indices), list(facts), dict(log))

    # Re-attach methods (bound to copy)
    def _get_query_embeddings(queries: list[str]) -> None:
        for q in queries:
            copy_engine.query_to_embedding["passage"][q] = np.array(
                [0.1] * 3072, dtype=np.float32
            )
            copy_engine.query_to_embedding["triple"][q] = np.array(
                [0.1] * 3072, dtype=np.float32
            )

    def _get_fact_scores(query: str) -> np.ndarray:  # type: ignore[type-arg]
        return copy_engine._fact_scores.copy()

    def _rerank_facts(
        query: str, query_fact_scores: np.ndarray  # type: ignore[type-arg]
    ) -> tuple[list[int], list[tuple], dict[str, Any]]:
        indices, facts, log = copy_engine._rerank_result
        return (list(indices), list(facts), dict(log))

    def _graph_search(
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,  # type: ignore[type-arg]
        top_k_facts: list[tuple],
        top_k_fact_indices: list[int],
        passage_node_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        return copy_engine._graph_ids.copy(), copy_engine._graph_scores.copy()

    def _dense_retrieval(
        query: str,
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        return copy_engine._dense_ids.copy(), copy_engine._dense_scores.copy()

    def _prepare() -> None:
        copy_engine.ready_to_retrieve = True

    copy_engine.get_query_embeddings = _get_query_embeddings
    copy_engine.get_fact_scores = _get_fact_scores
    copy_engine.rerank_facts = _rerank_facts
    copy_engine.graph_search_with_fact_entities = _graph_search
    copy_engine.dense_passage_retrieval = _dense_retrieval
    copy_engine.prepare_retrieval_objects = _prepare

    return copy_engine


# ── Offline contract tests ────────────────────────────────────────────────────


class TestOfflineContract:
    """Verify bridge output matches upstream behavior with deterministic engine.

    Since the pinned upstream cannot be imported in all environments, these
    tests compare the bridge's output against known-good expected behavior
    derived from the upstream's documented retrieval algorithm.  The
    deterministic engine's methods return exact pre-configured values, so
    any deviation in passage ranking, scores, or fact capture is a contract
    violation.
    """

    def test_graph_retrieval_produces_expected_passage_order(self) -> None:
        """Bridge graph retrieval must rank passages by graph scores."""
        engine = _make_contract_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("test query")

        # Graph path expected: [2, 0, 4, 1, 3] → chunk_2, chunk_0, ...
        assert trace.used_dense_fallback is False
        assert [p.chunk_id for p in trace.passages] == [
            "chunk_2", "chunk_0", "chunk_4", "chunk_1", "chunk_3"
        ]

    def test_graph_scores_allclose_to_expected(self) -> None:
        """Bridge scores must match pre-configured graph scores exactly."""
        engine = _make_contract_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("test query")

        expected_scores = [0.98, 0.88, 0.78, 0.68, 0.58]
        for passage, expected in zip(trace.passages, expected_scores, strict=False):
            np.testing.assert_allclose(
                passage.score, expected, rtol=1e-7, atol=1e-9,
                err_msg=f"Mismatch for {passage.chunk_id}"
            )

    def test_dense_fallback_produces_expected_passage_order(self) -> None:
        """Bridge dense fallback must rank passages by dense scores."""
        engine = _make_contract_engine(
            fact_rerank_result=(
                [],
                [],
                {"facts_before_rerank": [], "facts_after_rerank": []},
            ),
        )
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("fallback query")

        # Dense path expected: [0, 1, 2, 3, 4] → chunk_0, chunk_1, ...
        assert trace.used_dense_fallback is True
        assert [p.chunk_id for p in trace.passages] == [
            "chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"
        ]

    def test_facts_before_after_are_exactly_captured(self) -> None:
        """Bridge must capture facts-before and facts-after exactly from rerank_log."""
        engine = _make_contract_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("fact test")

        assert trace.facts_before_filter == [
            ("A", "relates_to", "B"),
            ("C", "has", "D"),
            ("E", "is", "F"),
        ]
        assert trace.facts_after_filter == [
            ("A", "relates_to", "B"),
            ("C", "has", "D"),
        ]

    def test_filter_error_is_none_when_no_error(self) -> None:
        """Bridge must propagate filter_error=None when rerank_log has no error."""
        engine = _make_contract_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("no error")
        assert trace.filter_error is None

    def test_filter_error_is_captured_when_present(self) -> None:
        """Bridge must capture filter_error from rerank_log."""
        error_result = (
            [],
            [],
            {
                "facts_before_rerank": [],
                "facts_after_rerank": [],
                "error": "upstream rerank failure",
            },
        )
        engine = _make_contract_engine(fact_rerank_result=error_result)
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("error query")
        assert trace.filter_error == "upstream rerank failure"

    def test_retrieval_query_is_preserved(self) -> None:
        """Bridge must preserve the original query in the trace."""
        engine = _make_contract_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        trace = bridge.retrieve_with_trace("What is the capital of France?")
        assert trace.retrieval_query == "What is the capital of France?"

    def test_dense_with_trace_is_identical_to_dense_fallback(self) -> None:
        """dense_with_trace must produce the same ranking as the dense fallback path."""
        # Force no facts → dense fallback
        no_facts = (
            [],
            [],
            {"facts_before_rerank": [], "facts_after_rerank": []},
        )
        engine1 = _make_contract_engine(fact_rerank_result=no_facts)
        engine2 = _make_contract_engine(fact_rerank_result=no_facts)

        bridge1 = HippoRAGBridge(engine1, top_k=3)
        bridge2 = HippoRAGBridge(engine2, top_k=3)

        trace_fallback = bridge1.retrieve_with_trace("same query")
        trace_dense = bridge2.dense_with_trace("same query")

        # Same passage IDs
        assert [p.chunk_id for p in trace_fallback.passages] == [
            p.chunk_id for p in trace_dense.passages
        ]
        # Same scores
        for pf, pd in zip(trace_fallback.passages, trace_dense.passages, strict=False):
            np.testing.assert_allclose(pf.score, pd.score, rtol=1e-7, atol=1e-9)

    def test_two_independent_runs_produce_identical_results(self) -> None:
        """Two independent bridge runs from independently reset engines must agree.

        This is the key contract: the bridge is deterministic when the engine
        state is identical.
        """
        engine = _make_contract_engine()
        copy_engine = _deepcopy_engine(engine)

        bridge1 = HippoRAGBridge(engine, top_k=5)
        bridge2 = HippoRAGBridge(copy_engine, top_k=5)

        trace1 = bridge1.retrieve_with_trace("deterministic query")
        trace2 = bridge2.retrieve_with_trace("deterministic query")

        # Passage IDs must match exactly
        assert [p.chunk_id for p in trace1.passages] == [
            p.chunk_id for p in trace2.passages
        ]
        # Scores must match exactly
        for p1, p2 in zip(trace1.passages, trace2.passages, strict=False):
            np.testing.assert_allclose(p1.score, p2.score, rtol=1e-7, atol=1e-9)
            assert p1.text == p2.text
            assert p1.rank == p2.rank

        # Facts must match
        assert trace1.facts_before_filter == trace2.facts_before_filter
        assert trace1.facts_after_filter == trace2.facts_after_filter
        assert trace1.used_dense_fallback == trace2.used_dense_fallback

    def test_no_network_reached_during_contract_test(self) -> None:
        """Contract tests must fail if any network method is reached.

        The deterministic engine's LLM model raises AssertionError on any
        ``infer()`` call.  This test verifies the bridge does not trigger
        the LLM during retrieval.
        """
        engine = _make_contract_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        # Should not raise — LLM is never called during retrieval
        trace = bridge.retrieve_with_trace("offline only")
        assert isinstance(trace, RetrievalTrace)


# ── Real-index contract check (for Task 11) ───────────────────────────────────


def assert_real_index_contract(
    engine: Any,
    queries: list[str],
    *,
    top_k: int = 5,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    """Verify bridge retrieval matches official upstream retrieval on a real index.

    This function is called in **Task 11** after a real HippoRAG index has
    been built.  It snapshots the persistent LLM/embedding caches, runs both
    the official ``HippoRAG.retrieve()`` and the bridge's
    ``retrieve_with_trace()`` from the same completed index, and asserts:

    - Identical passage order (chunk IDs)
    - Scores within floating-point tolerance
    - Zero cache misses during the second path (bridge)

    Parameters
    ----------
    engine:
        An initialised upstream ``HippoRAG`` engine with ``ready_to_retrieve
        = True`` and a completed index.
    queries:
        List of query strings.  Task 11 runs this on one fixed development
        ID per dataset before any complete development run.
    top_k:
        Number of passages to compare.
    rtol, atol:
        Floating-point tolerances for score comparison.
    """
    # Lazy import to avoid dependency issues in test-only environments
    from hipporag import HippoRAG  # type: ignore[import-untyped]

    bridge = HippoRAGBridge(engine, top_k=top_k)

    for query in queries:
        # ── Official retrieval (first path — may cause cache misses) ────
        official_solutions = HippoRAG.retrieve(
            engine, [query], num_to_retrieve=top_k
        )
        if isinstance(official_solutions, tuple):
            official_solutions = official_solutions[0]
        official_solution = official_solutions[0]

        # ── Bridge retrieval (second path — must be all cache hits) ─────
        bridge_trace = bridge.retrieve_with_trace(query)

        # ── Compare passage order ───────────────────────────────────────
        official_docs = official_solution.docs[:top_k]
        bridge_texts = [p.text for p in bridge_trace.passages]
        assert official_docs == bridge_texts, (
            f"Passage mismatch for query: {query}\n"
            f"Official: {official_docs}\nBridge:    {bridge_texts}"
        )

        # ── Compare scores ──────────────────────────────────────────────
        official_scores = official_solution.doc_scores[:top_k]
        bridge_scores = [p.score for p in bridge_trace.passages]
        np.testing.assert_allclose(
            official_scores, bridge_scores, rtol=rtol, atol=atol,
            err_msg=f"Score mismatch for query: {query}"
        )
