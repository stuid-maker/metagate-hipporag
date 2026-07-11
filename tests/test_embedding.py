"""Tests for PersistentOpenAIEmbeddingModel — SQLite cache, ledger, and preprocessing.

All tests use a fake embedding backend so they run offline without an API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from metagate_hipporag.embedding import (
    PersistentOpenAIEmbeddingModel,
    _embedding_cache_key,
    export_query_embeddings,
    inject_embedding_model,
)
from metagate_hipporag.provenance import UsageLedger

# ── Fake backend ─────────────────────────────────────────────────────────────

class FakeEmbeddingResponse:
    """Minimal fake OpenAI embeddings response."""

    def __init__(self, embeddings: list[list[float]], tokens: int = 0):
        self.data = [
            type("_Emb", (), {"embedding": emb})() for emb in embeddings
        ]
        self.usage = type("_Usage", (), {"total_tokens": tokens})()


class FakeOpenAIClient:
    """Records calls and returns pre-configured embedding vectors."""

    def __init__(self, vectors: list[list[float]] | None = None):
        self.calls: list[dict[str, Any]] = []
        self._vectors = vectors or [[0.1] * 3072]

    @property
    def embeddings(self) -> FakeOpenAIClient:
        return self

    def create(self, *, input: list[str], model: str, **kwargs: Any) -> FakeEmbeddingResponse:
        actual = kwargs.get("dimensions", 3072)
        self.calls.append({"input": input, "model": model, "dimensions": actual})
        if len(self._vectors) < len(input):
            # Extend with copies of the first vector
            self._vectors.extend([self._vectors[0]] * (len(input) - len(self._vectors)))
        return FakeEmbeddingResponse(self._vectors[: len(input)], tokens=len(input) * 10)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fake_global_config() -> Any:
    """Build a minimal fake global config that the upstream __init__ accepts."""

    class FakeGlobalConfig:
        embedding_model_name = "text-embedding-3-large"
        embedding_return_as_normalized = True  # emulate norm=True
        embedding_max_seq_len = 8192
        embedding_batch_size = 64
        embedding_base_url = "https://api.openai.com/v1"
        azure_embedding_endpoint = None

        def __getattr__(self, name: str) -> Any:
            # Return defaults for any access the upstream might try
            return None

    return FakeGlobalConfig()


def _make_model(
    tmp_path: Path,
    *,
    dimensions: int = 3072,
    instruction_mode: str = "upstream_ignored",
    ledger: UsageLedger | None = None,
    fake_vectors: list[list[float]] | None = None,
) -> tuple[PersistentOpenAIEmbeddingModel, FakeOpenAIClient]:
    """Construct a model with a fake OpenAI client, resolving args correctly.

    The upstream ``OpenAIEmbeddingModel.__init__`` calls ``_init_embedding_config()``
    which inspects ``self.global_config``.  We pass a minimal config object.
    """
    fake_client = FakeOpenAIClient(fake_vectors)
    model = PersistentOpenAIEmbeddingModel.__new__(PersistentOpenAIEmbeddingModel)
    # Bypass the full upstream __init__ — we set attributes manually.
    model.global_config = _fake_global_config()
    model.embedding_model_name = "text-embedding-3-large"
    model._init_embedding_config()  # sets embedding_config, client
    model.client = fake_client  # type: ignore[attr-defined]
    model._cache_db_path = tmp_path / "embedding_cache.sqlite"
    model._dimensions = dimensions
    model._instruction_mode = instruction_mode
    model._ledger = ledger
    model._price_per_million = 0.13
    model._project_limit_usd = 18.0
    model.embedding_dim = dimensions
    model._init_cache_db()
    return model, fake_client


# ── Cache key tests ──────────────────────────────────────────────────────────


def test_cache_key_is_deterministic() -> None:
    k1 = _embedding_cache_key("text-embedding-3-large", 3072, "hello", "upstream_ignored")
    k2 = _embedding_cache_key("text-embedding-3-large", 3072, "hello", "upstream_ignored")
    assert k1 == k2
    assert len(k1) == 64


def test_cache_key_changes_with_model() -> None:
    k1 = _embedding_cache_key("text-embedding-3-large", 3072, "hello", "upstream_ignored")
    k2 = _embedding_cache_key("text-embedding-3-small", 3072, "hello", "upstream_ignored")
    assert k1 != k2


def test_cache_key_changes_with_dimensions() -> None:
    k1 = _embedding_cache_key("model", 3072, "hello", "upstream_ignored")
    k2 = _embedding_cache_key("model", 1536, "hello", "upstream_ignored")
    assert k1 != k2


def test_cache_key_changes_with_text() -> None:
    k1 = _embedding_cache_key("model", 3072, "hello", "upstream_ignored")
    k2 = _embedding_cache_key("model", 3072, "world", "upstream_ignored")
    assert k1 != k2


def test_cache_key_changes_with_instruction_mode() -> None:
    k1 = _embedding_cache_key("model", 3072, "hello", "upstream_ignored")
    k2 = _embedding_cache_key("model", 3072, "hello", "query")
    assert k1 != k2


# ── encode / batch_encode tests ──────────────────────────────────────────────


def test_encode_returns_float32_unnormalized(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    result = model.encode(["test text"])
    assert result.dtype == np.float32
    assert result.shape == (1, 3072)


def test_identical_text_calls_backend_once(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.encode(["alpha", "alpha"])
    assert len(fake.calls) == 1
    # Only "alpha" appears once — duplicates within a single batch share the call
    assert fake.calls[0]["input"] == ["alpha"]


def test_different_texts_call_backend(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.encode(["alpha", "beta"])
    assert len(fake.calls) == 1
    assert set(fake.calls[0]["input"]) == {"alpha", "beta"}


def test_encode_cross_call_cache_hit(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.encode(["alpha"])
    assert len(fake.calls) == 1
    model.encode(["alpha", "beta"])
    # "alpha" should be cached, only "beta" goes to the backend
    assert len(fake.calls) == 2
    assert fake.calls[1]["input"] == ["beta"]


def test_batch_encode_batches_and_normalizes(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.embedding_config.norm = True
    vec = [0.0] * 3072
    vec[0] = 4.0  # nonzero
    model, fake = _make_model(tmp_path, fake_vectors=[vec])
    model.embedding_config.norm = True
    result = model.batch_encode(["text"])
    # Should be L2-normalized
    norm = np.linalg.norm(result[0])
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_batch_encode_chunks_at_64(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.embedding_config.norm = False
    model.embedding_config.encode_params["batch_size"] = 64
    texts = [f"text-{i}" for i in range(130)]
    result = model.batch_encode(texts)
    assert result.shape == (130, 3072)
    # Should have made ceil(130/64) = 3 API calls
    assert len(fake.calls) == 3


def test_encode_preprocesses_newline_to_space(tmp_path: Path) -> None:
    """Upstream replaces ``\\n`` with space, empty with single space."""
    model, fake = _make_model(tmp_path)
    model.encode(["a\nb"])
    assert fake.calls[0]["input"] == ["a b"]


def test_encode_preprocesses_empty_to_space(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.encode([""])
    assert fake.calls[0]["input"] == [" "]


def test_encode_preprocesses_empty_after_newline_replacement(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    model.encode(["\n"])
    # "\n" → "" (after replace) → " "
    assert fake.calls[0]["input"] == [" "]


def test_embedding_dim_is_set(tmp_path: Path) -> None:
    model, fake = _make_model(tmp_path)
    assert model.embedding_dim == 3072


# ── __version__ import ──────────────────────────────────────────────────────


def test_version_is_string() -> None:
    from metagate_hipporag import __version__

    assert isinstance(__version__, str)
    assert "." in __version__


# ── Ledger integration ───────────────────────────────────────────────────────


def _make_ledger(tmp_path: Path) -> UsageLedger:
    return UsageLedger(tmp_path / "ledger.db", limit_usd=100.0)


def test_cache_hit_no_ledger_entry(tmp_path: Path) -> None:
    """Cache hits should not generate ledger entries (already settled on miss)."""
    ledger = _make_ledger(tmp_path)
    model, fake = _make_model(tmp_path, ledger=ledger)
    before = ledger.snapshot()
    model.encode(["text-A"])
    after_first = ledger.snapshot()
    # One call → embedding_tokens should increase
    assert after_first.embedding_tokens > before.embedding_tokens

    model.encode(["text-A"])  # cache hit
    after_second = ledger.snapshot()
    # No new tokens — embedding_tokens unchanged
    assert after_second.embedding_tokens == after_first.embedding_tokens
    assert after_second.actual_usd == after_first.actual_usd


def test_cache_miss_records_usage(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    model, fake = _make_model(tmp_path, ledger=ledger)
    before = ledger.snapshot()
    model.encode(["text-A", "text-B"])
    after = ledger.snapshot()
    assert after.embedding_tokens > before.embedding_tokens
    assert after.actual_usd > before.actual_usd


def test_main_experiment_uses_3072_dimensions_upstream_ignored() -> None:
    """Smoke test: verify the config defaults match what Task 6 requires."""
    from metagate_hipporag.config import load_config

    config = load_config(Path("configs/experiment.yaml"))
    assert config.models.embedding_dimensions == 3072
    assert config.models.embedding_instruction_mode == "upstream_ignored"
    assert config.models.embedding == "text-embedding-3-large"


# ── inject / export tests ───────────────────────────────────────────────────


class FakeEmbeddingStore:
    """Minimal fake for engine embedding stores."""

    def __init__(self) -> None:
        self.embedding_model: Any = None


class FakeHippoRAGEngine:
    """Minimal fake HippoRAG engine for injection tests."""

    def __init__(self) -> None:
        self.embedding_model: Any = None
        self.chunk_embedding_store = FakeEmbeddingStore()
        self.entity_embedding_store = FakeEmbeddingStore()
        self.fact_embedding_store = FakeEmbeddingStore()
        self.query_to_embedding: dict[str, np.ndarray] = {}


def test_inject_embedding_model_sets_all_stores() -> None:
    engine = FakeHippoRAGEngine()
    dummy = object()
    inject_embedding_model(engine, dummy)
    assert engine.embedding_model is dummy
    assert engine.chunk_embedding_store.embedding_model is dummy
    assert engine.entity_embedding_store.embedding_model is dummy
    assert engine.fact_embedding_store.embedding_model is dummy


def test_export_query_embeddings_writes_npz(tmp_path: Path) -> None:
    engine = FakeHippoRAGEngine()
    engine.query_to_embedding["q1"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    engine.query_to_embedding["q2"] = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    target = tmp_path / "queries.npz"
    export_query_embeddings(engine, target)
    assert target.exists()
    loaded = np.load(target)
    assert set(loaded.keys()) == {"q1", "q2"}
    np.testing.assert_array_equal(loaded["q1"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(loaded["q2"], np.array([4.0, 5.0, 6.0]))


# ── Index fingerprint tests ──────────────────────────────────────────────────


def test_index_config_hash_is_deterministic() -> None:
    from metagate_hipporag.provenance import index_config_hash

    p: dict[str, object] = {
        "corpus_sha256": "a" * 64,
        "upstream_sha": "b" * 40,
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
    }
    first = index_config_hash(**p)
    second = index_config_hash(**p)
    assert first == second
    assert len(first) == 64


def test_index_config_hash_ignores_unknown_keys() -> None:
    from metagate_hipporag.provenance import index_config_hash

    p: dict[str, object] = {
        "corpus_sha256": "a" * 64,
        "gate_threshold": 0.75,
        "sampling_dev_size": 100,
        "pricing_usd": 0.13,
    }
    # Unknown keys are ignored; only corpus_sha256 contributes
    h1 = index_config_hash(**p)
    h2 = index_config_hash(corpus_sha256="a" * 64)
    assert h1 == h2


def test_index_config_hash_rejects_empty() -> None:
    from metagate_hipporag.provenance import index_config_hash

    try:
        index_config_hash()
    except ValueError:
        pass
    else:
        raise AssertionError("empty params were accepted")


def test_index_config_hash_changes_with_corpus() -> None:
    from metagate_hipporag.provenance import index_config_hash

    h1 = index_config_hash(corpus_sha256="a" * 64, upstream_sha="b" * 40)
    h2 = index_config_hash(corpus_sha256="c" * 64, upstream_sha="b" * 40)
    assert h1 != h2


def test_index_directory_layout() -> None:
    from metagate_hipporag.provenance import index_directory

    dirpath = index_directory(
        dataset="musique",
        corpus_sha256="a" * 64,
        upstream_sha="b" * 40,
        llm_slug="gpt-4o-mini",
        embedding_slug="text-embedding-3-large",
        openie_prompt_sha256="c" * 64,
        index_config_sha256="d" * 64,
        base=Path("/tmp/test-indexes"),
    )
    parts = dirpath.parts
    # Last 6 segments should be: dataset, corpus_sha12, upstream_sha12,
    #   llm_slug, embedding_slug, openie_prompt_sha12, index_config_sha12
    assert parts[-7] == "musique"
    assert len(parts[-6]) == 12  # corpus_sha12
    assert len(parts[-5]) == 12  # upstream_sha12
    assert parts[-4] == "gpt-4o-mini"
    assert parts[-3] == "text-embedding-3-large"
    assert len(parts[-2]) == 12  # openie_prompt_sha12
    assert len(parts[-1]) == 12  # index_config_sha12


def test_index_directory_default_base() -> None:
    from metagate_hipporag.provenance import index_directory

    dirpath = index_directory(
        dataset="nq_rear",
        corpus_sha256="0" * 64,
        upstream_sha="0" * 40,
        llm_slug="gpt-4o-mini",
        embedding_slug="text-embedding-3-large",
        openie_prompt_sha256="0" * 64,
        index_config_sha256="0" * 64,
    )
    # Default base should contain 'artifacts/indexes'
    assert "artifacts" in str(dirpath)
    assert "indexes" in str(dirpath)
