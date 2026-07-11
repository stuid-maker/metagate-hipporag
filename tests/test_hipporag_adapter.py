"""Tests for HippoRAGBridge — unit tests with a deterministic fake engine.

Covers:
- Graph retrieval path (facts present → graph_search_with_fact_entities)
- Empty-filter dense fallback (no facts → dense_passage_retrieval)
- dense_with_trace (dense-only retrieval)
- answer (QA via upstream engine.qa)
- Chunk-ID mapping, top-k truncation, facts before/after capture
- Original query preservation in RetrievalTrace
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metagate_hipporag.models import RetrievalTrace, RetrievedPassage

# ── Fake engine helpers ───────────────────────────────────────────────────────

class FakeGlobalConfig:
    """Minimal fake upstream global config."""

    linking_top_k: int = 5
    passage_node_weight: float = 0.05
    dataset: str = "musique"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeChunkEmbeddingStore:
    """Fake chunk embedding store with in-memory rows."""

    def __init__(self, rows: dict[str, dict[str, Any]] | None = None) -> None:
        self._rows: dict[str, dict[str, Any]] = rows or {}

    def get_row(self, chunk_id: str) -> dict[str, Any]:
        return self._rows.get(chunk_id, {"content": f"content of {chunk_id}"})

    def get_rows(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {cid: self.get_row(cid) for cid in chunk_ids}

    def get_all_ids(self) -> list[str]:
        return list(self._rows.keys())


class FakeLLMModel:
    """Fake LLM model that records calls and returns canned responses."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def infer(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any], bool]:
        self.calls.append(messages)
        return ("fake answer", {"model": "fake"}, False)


class FakeRerankFilter:
    """Fake rerank filter with replaceable infer function."""

    def __init__(self) -> None:
        self.llm_infer_fn = None


class FakeEmbeddingStore(FakeChunkEmbeddingStore):
    """Shared store base for fact/entity stores."""


def make_fake_engine(
    *,
    passage_node_keys: list[str] | None = None,
    passage_contents: dict[str, str] | None = None,
    fact_node_keys: list[str] | None = None,
    fact_scores: np.ndarray | None = None,
    dense_doc_ids: np.ndarray | None = None,
    dense_doc_scores: np.ndarray | None = None,
    graph_doc_ids: np.ndarray | None = None,
    graph_doc_scores: np.ndarray | None = None,
    rerank_result: tuple[list[int], list[tuple], dict] | None = None,
    ready: bool = False,
    global_config_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Construct a deterministic fake upstream engine.

    All retrieval paths return pre-configured values so tests are offline and
    reproducible.  Unspecified attributes get sensible defaults.
    """
    engine = type("_FakeEngine", (), {})()

    # Default passage info
    if passage_node_keys is None:
        passage_node_keys = ["p0", "p1", "p2", "p3", "p4"]
    if passage_contents is None:
        passage_contents = {key: f"Text of {key}" for key in passage_node_keys}

    store_rows = {key: {"content": passage_contents.get(key, f"Text of {key}")}
                  for key in passage_node_keys}
    engine.chunk_embedding_store = FakeChunkEmbeddingStore(store_rows)

    # Fact-related
    engine.fact_node_keys = fact_node_keys or ["f0", "f1", "f2"]
    engine.entity_node_keys = ["e0", "e1"]
    engine.entity_embedding_store = FakeEmbeddingStore()
    engine.fact_embedding_store = FakeEmbeddingStore()

    # Retrieval state
    engine.ready_to_retrieve = ready
    engine.query_to_embedding: dict[str, dict[str, np.ndarray]] = {
        "triple": {},
        "passage": {},
    }

    # Pre-configured returns
    engine._dense_doc_ids = (
        dense_doc_ids if dense_doc_ids is not None
        else np.array([0, 1, 2, 3, 4])
    )
    engine._dense_doc_scores = (
        dense_doc_scores if dense_doc_scores is not None
        else np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    )
    engine._graph_doc_ids = (
        graph_doc_ids if graph_doc_ids is not None
        else np.array([2, 0, 4, 1, 3])
    )
    engine._graph_doc_scores = (
        graph_doc_scores if graph_doc_scores is not None
        else np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    )
    engine._fact_scores = fact_scores if fact_scores is not None else np.array([0.8, 0.6, 0.4])
    engine._rerank_result = rerank_result or (
        [0, 1],
        [("e0", "relates_to", "e1")],
        {
            "facts_before_rerank": [("e0", "relates_to", "e1"), ("e2", "has", "e3")],
            "facts_after_rerank": [("e0", "relates_to", "e1")],
        },
    )

    # Global config
    engine.global_config = FakeGlobalConfig(**(global_config_kwargs or {}))

    # LLM + rerank
    engine.llm_model = FakeLLMModel()
    engine.rerank_filter = FakeRerankFilter()

    # Passage node keys (list of chunk IDs)
    engine.passage_node_keys = passage_node_keys

    # ── Replaceable methods (to match upstream API) ────────────────────

    def _prepare_retrieval_objects() -> None:
        engine.ready_to_retrieve = True
        engine.passage_node_keys = passage_node_keys
        engine.fact_node_keys = fact_node_keys or []

    def _get_query_embeddings(queries: list[str]) -> None:
        for q in queries:
            engine.query_to_embedding["passage"][q] = np.array(
                [0.1] * 3072, dtype=np.float32
            )
            engine.query_to_embedding["triple"][q] = np.array(
                [0.1] * 3072, dtype=np.float32
            )

    def _get_fact_scores(query: str) -> np.ndarray:  # type: ignore[type-arg]
        return engine._fact_scores

    def _rerank_facts(
        query: str, query_fact_scores: np.ndarray  # type: ignore[type-arg]
    ) -> tuple[list[int], list[tuple], dict[str, Any]]:
        return engine._rerank_result

    def _graph_search_with_fact_entities(
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,  # type: ignore[type-arg]
        top_k_facts: list[tuple],
        top_k_fact_indices: list[int],
        passage_node_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        return engine._graph_doc_ids, engine._graph_doc_scores

    def _dense_passage_retrieval(
        query: str,
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        return engine._dense_doc_ids, engine._dense_doc_scores

    def _qa(
        queries: list[Any],
    ) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
        response_messages: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        for qs in queries:
            msg, meta, _ = engine.llm_model.infer(
                [{"role": "user", "content": qs.question}]
            )
            response_messages.append(msg)
            metadata_list.append(meta)
        return queries, response_messages, metadata_list

    engine.prepare_retrieval_objects = _prepare_retrieval_objects
    engine.get_query_embeddings = _get_query_embeddings
    engine.get_fact_scores = _get_fact_scores
    engine.rerank_facts = _rerank_facts
    engine.graph_search_with_fact_entities = _graph_search_with_fact_entities
    engine.dense_passage_retrieval = _dense_passage_retrieval
    engine.qa = _qa

    return engine


# ── Helpers ───────────────────────────────────────────────────────────────────


def p(chunk_id: str, score: float, rank: int, text: str | None = None) -> RetrievedPassage:
    """Shorthand for constructing RetrievedPassage in assertions."""
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=text or f"Text of {chunk_id}",
        score=score,
        rank=rank,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestHippoRAGBridgeInit:
    """Test bridge initialisation."""

    def test_stores_engine_and_top_k(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine()
        bridge = HippoRAGBridge(engine, top_k=5)
        assert bridge.engine is engine
        assert bridge.top_k == 5

    def test_default_top_k_is_5(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine()
        bridge = HippoRAGBridge(engine)
        assert bridge.top_k == 5


class TestRetrieveWithTrace:
    """Test the core graph retrieval path."""

    def test_graph_path_with_facts_returns_passages_and_trace(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine, top_k=3)
        trace = bridge.retrieve_with_trace("What is X?")

        # Should use graph path
        assert trace.used_dense_fallback is False
        assert trace.filter_error is None

        # Check passages
        assert len(trace.passages) == 3
        assert trace.passages[0].chunk_id == "p2"  # top graph result
        assert trace.passages[0].rank == 1
        assert trace.passages[1].chunk_id == "p0"
        assert trace.passages[1].rank == 2
        assert trace.passages[2].chunk_id == "p4"
        assert trace.passages[2].rank == 3

        # Check facts captured
        assert len(trace.facts_before_filter) == 2
        assert len(trace.facts_after_filter) == 1

        # Original query preserved
        assert trace.retrieval_query == "What is X?"

    def test_dense_fallback_when_no_facts(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        # Empty facts → rerank returns empty
        engine = make_fake_engine(
            ready=True,
            rerank_result=([], [], {"facts_before_rerank": [], "facts_after_rerank": []}),
        )
        bridge = HippoRAGBridge(engine, top_k=3)
        trace = bridge.retrieve_with_trace("fallback query")

        assert trace.used_dense_fallback is True
        # Dense fallback uses pre-configured dense doc IDs
        assert trace.passages[0].chunk_id == "p0"

    def test_top_k_truncation_respected(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine, top_k=2)
        trace = bridge.retrieve_with_trace("query")

        assert len(trace.passages) == 2
        assert trace.passages[0].rank == 1
        assert trace.passages[1].rank == 2

    def test_chunk_id_mapping_is_correct(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(
            ready=True,
            passage_node_keys=["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"],
            passage_contents={
                "chunk_a": "Alpha",
                "chunk_b": "Beta",
                "chunk_c": "Gamma",
                "chunk_d": "Delta",
                "chunk_e": "Epsilon",
            },
            dense_doc_ids=np.array([2, 0, 4, 1, 3]),
            dense_doc_scores=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
        )
        # Use dense fallback to test chunk ID mapping
        engine._rerank_result = ([], [], {"facts_before_rerank": [], "facts_after_rerank": []})
        bridge = HippoRAGBridge(engine, top_k=3)
        trace = bridge.retrieve_with_trace("map test")

        assert trace.passages[0].chunk_id == "chunk_c"  # doc_id=2
        assert trace.passages[0].text == "Gamma"
        assert trace.passages[1].chunk_id == "chunk_a"  # doc_id=0
        assert trace.passages[1].text == "Alpha"
        assert trace.passages[2].chunk_id == "chunk_e"  # doc_id=4
        assert trace.passages[2].text == "Epsilon"

    def test_prepare_retrieval_objects_called_when_not_ready(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=False)
        bridge = HippoRAGBridge(engine)
        trace = bridge.retrieve_with_trace("query")
        # Should not raise; prepare_retrieval_objects called internally
        assert isinstance(trace, RetrievalTrace)

    def test_filter_error_captured(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        rerank_with_error = (
            [],
            [],
            {
                "facts_before_rerank": [],
                "facts_after_rerank": [],
                "error": "some upstream error",
            },
        )
        engine = make_fake_engine(ready=True, rerank_result=rerank_with_error)
        bridge = HippoRAGBridge(engine)
        trace = bridge.retrieve_with_trace("error query")
        assert trace.filter_error == "some upstream error"


class TestDenseWithTrace:
    """Test the dense-only retrieval path."""

    def test_dense_with_trace_uses_dense_path(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine, top_k=3)
        trace = bridge.dense_with_trace("dense only")

        assert trace.used_dense_fallback is True
        assert len(trace.passages) == 3
        assert trace.passages[0].chunk_id == "p0"  # top dense result
        # Facts should be empty (dense path skips fact extraction)
        assert trace.facts_before_filter == []
        assert trace.facts_after_filter == []

    def test_dense_with_trace_skips_fact_rerank(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine)
        _trace = bridge.dense_with_trace("query")

        # LLM model should not have been called (no rerank)
        assert engine.llm_model.calls == []


class TestAnswer:
    """Test the QA bridge method."""

    def test_answer_constructs_query_solution_and_calls_qa(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine)

        passages = [
            RetrievedPassage(chunk_id="p0", text="Passage 0", score=0.9, rank=1),
            RetrievedPassage(chunk_id="p1", text="Passage 1", score=0.7, rank=2),
        ]
        result = bridge.answer("Who discovered gravity?", passages)

        # Returns (query_solutions, response_messages, metadata) like upstream
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[1] == ["fake answer"]  # response messages
        assert result[0][0].question == "Who discovered gravity?"

    def test_answer_records_llm_usage(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine)

        passages = [RetrievedPassage(chunk_id="p0", text="Passage 0", score=1.0, rank=1)]
        _result = bridge.answer("Question?", passages)

        # The fake LLM was called
        assert len(engine.llm_model.calls) == 1


class TestUsageSnapshot:
    """Test that bridge methods can record usage snapshots."""

    def test_retrieve_with_trace_includes_usage_field(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine)
        trace = bridge.retrieve_with_trace("usage test")

        assert trace.usage is not None
        assert trace.usage.observed_latency_seconds > 0.0
        assert trace.usage.method_equivalent_latency_seconds > 0.0

    def test_dense_with_trace_includes_usage_field(self) -> None:
        from metagate_hipporag.hipporag_adapter import HippoRAGBridge

        engine = make_fake_engine(ready=True)
        bridge = HippoRAGBridge(engine)
        trace = bridge.dense_with_trace("usage test")

        assert trace.usage is not None
        assert trace.usage.observed_latency_seconds > 0.0


class TestInferenceWrapper:
    """Test the upstream LLM inference wrapper for usage snapshots."""

    def test_wrapper_records_inference_call_and_returns_upstream_tuple(self) -> None:
        from metagate_hipporag.hipporag_adapter import wrap_upstream_inference

        engine = make_fake_engine(ready=True)
        # Record calls
        calls: list[tuple[str, dict[str, Any], bool]] = []

        def recorder(
            response: str, metadata: dict[str, Any], cache_hit: bool
        ) -> None:
            calls.append((response, metadata, cache_hit))

        wrapper = wrap_upstream_inference(engine, on_call=recorder)

        result = wrapper([{"role": "user", "content": "test"}])
        assert result == ("fake answer", {"model": "fake"}, False)
        assert len(calls) == 1
        assert calls[0][0] == "fake answer"

    def test_wrapper_redirects_rerank_filter_llm_infer_fn(self) -> None:
        from metagate_hipporag.hipporag_adapter import wrap_upstream_inference

        engine = make_fake_engine(ready=True)

        _wrapper = wrap_upstream_inference(engine)
        # The rerank filter's llm_infer_fn should be set to the wrapper
        assert engine.rerank_filter.llm_infer_fn is not None
        # Calling it should return the upstream-format tuple
        result = engine.rerank_filter.llm_infer_fn([{"role": "user", "content": "q"}])
        assert isinstance(result, tuple)
        assert len(result) == 3  # (response, metadata, cache_hit)
