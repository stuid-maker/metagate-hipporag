"""Persistent OpenAI embedding cache with SQLite storage and usage ledger.

Provides ``PersistentOpenAIEmbeddingModel``, a subclass of the pinned upstream
``OpenAIEmbeddingModel`` that intercepts ``encode()`` and ``batch_encode()`` to
cache embedding vectors in SQLite.  Every cache miss is recorded in the
``UsageLedger``.

Also provides ``inject_embedding_model()`` to install the model into a HippoRAG
engine before indexing / retrieval, and ``export_query_embeddings()`` to persist
query vectors for resumable runs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .models import LedgerEntry
from .provenance import UsageLedger

# ── Cache key ────────────────────────────────────────────────────────────────


def _embedding_cache_key(
    model: str,
    dimensions: int,
    raw_text: str,
    instruction_mode: str,
) -> str:
    """SHA-256 hex digest of the canonical embedding request identity.

    The key is computed from the *raw* text before upstream preprocessing
    (``\n`` → `` ``, empty → `` ``) so that it is stable across re-runs.
    """
    payload = {
        "model": model,
        "dimensions": dimensions,
        "raw_text": raw_text,
        "instruction_mode": instruction_mode,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Preprocessing (pinned upstream) ─────────────────────────────────────────


def _preprocess(texts: list[str]) -> list[str]:
    """Exactly reproduce pinned upstream OpenAIEmbeddingModel.encode() preprocessing.

    1. Replace every newline with one space.
    2. Replace an empty resulting string with one space.
    """
    result: list[str] = []
    for t in texts:
        t = t.replace("\n", " ")
        if t == "":
            t = " "
        result.append(t)
    return result


# ── Persistent model ─────────────────────────────────────────────────────────


class PersistentOpenAIEmbeddingModel:
    """Persistent OpenAI embedding model with SQLite cache and usage ledger.

    Subclass of the pinned upstream ``OpenAIEmbeddingModel`` that intercepts
    ``encode()`` and ``batch_encode()``.  Cached vectors survive across process
    restarts.  Every cache miss is recorded as a ``LedgerEntry``.
    """

    # Set by the factory so test harnesses can construct manually.
    global_config: Any
    embedding_model_name: str
    embedding_config: Any
    client: Any
    embedding_dim: int

    def __init__(
        self,
        global_config: Any,
        *,
        cache_db_path: Path,
        dimensions: int,
        instruction_mode: str,
        ledger: UsageLedger | None = None,
        price_per_million: float = 0.13,
        project_limit_usd: float = 18.0,
    ) -> None:
        """Initialise the model.

        Parameters
        ----------
        global_config:
            Upstream ``BaseConfig``-compatible object with at least
            ``embedding_model_name``, ``embedding_return_as_normalized``,
            ``embedding_max_seq_len``, ``embedding_batch_size``,
            ``embedding_base_url``, and ``azure_embedding_endpoint``.
        """
        self.global_config = global_config
        self.embedding_model_name = global_config.embedding_model_name
        self._cache_db_path = cache_db_path
        self._dimensions = dimensions
        self._instruction_mode = instruction_mode
        self._ledger = ledger
        self._price_per_million = price_per_million
        self._project_limit_usd = project_limit_usd
        self.embedding_dim = dimensions

        # Initialise the upstream embedding config and OpenAI client.
        self._init_embedding_config()
        self.client = self._create_openai_client()
        self._init_cache_db()

    # ── Upstream-compatible initialisation ──────────────────────────────

    def _init_embedding_config(self) -> None:
        """Replicate the upstream ``_init_embedding_config()``.

        Uses the same attribute paths as the pinned upstream so that
        ``batch_encode()`` can rely on ``self.embedding_config.norm`` etc.
        """
        # Import the upstream EmbeddingConfig (lazy — the patched __init__
        # only imports this module when the model name matches).
        from hipporag.embedding_model.base import EmbeddingConfig  # type: ignore[import-untyped]

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "device_map": "auto",
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

    def _create_openai_client(self) -> Any:
        """Create an OpenAI client mirroring upstream logic."""
        from openai import AzureOpenAI, OpenAI

        if self.global_config.azure_embedding_endpoint is None:
            return OpenAI(base_url=self.global_config.embedding_base_url)
        api_version = self.global_config.azure_embedding_endpoint.split("api-version=")[1]
        return AzureOpenAI(
            api_version=api_version,
            azure_endpoint=self.global_config.azure_embedding_endpoint,
        )

    def _init_cache_db(self) -> None:
        """Create the SQLite cache table if it does not exist."""
        self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._cache_db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "  key TEXT PRIMARY KEY,"
                "  model TEXT NOT NULL,"
                "  text_sha256 TEXT NOT NULL,"
                "  vector BLOB NOT NULL,"
                "  dimensions INTEGER NOT NULL"
                ")"
            )
            conn.commit()

    # ── Cache primitives ────────────────────────────────────────────────

    def _lookup(self, key: str) -> np.ndarray | None:  # type: ignore[type-arg]
        """Return cached float32 vector or *None*."""
        with sqlite3.connect(str(self._cache_db_path)) as conn:
            row = conn.execute(
                "SELECT vector, dimensions FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        blob, dims = row
        return np.frombuffer(blob, dtype=np.float32).reshape(dims).copy()

    def _store(self, key: str, raw_text: str, vector: np.ndarray) -> None:  # type: ignore[type-arg]
        """Persist a single vector to the cache."""
        text_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        vector_f32 = np.asarray(vector, dtype=np.float32)
        with sqlite3.connect(str(self._cache_db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(key, model, text_sha256, vector, dimensions) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    self.embedding_model_name,
                    text_sha256,
                    vector_f32.tobytes(),
                    int(vector_f32.shape[0]),
                ),
            )
            conn.commit()

    # ── API call + ledger ───────────────────────────────────────────────

    def _call_api(self, preprocessed: list[str]) -> tuple[np.ndarray, int]:  # type: ignore[type-arg]
        """Call the real OpenAI embeddings endpoint and return (vectors, total_tokens)."""
        response = self.client.embeddings.create(
            input=preprocessed,
            model=self.embedding_model_name,
            dimensions=self._dimensions,
        )
        vectors = np.array([v.embedding for v in response.data], dtype=np.float32)
        tokens = response.usage.total_tokens
        return vectors, tokens

    # ── encode (unnormalized, cached) ───────────────────────────────────

    def encode(self, texts: list[str]) -> np.ndarray:  # type: ignore[type-arg]
        """Return unnormalised float32 embedding rows.

        Cached texts are returned from SQLite; missed texts are fetched from
        the OpenAI API in a single batch call and then persisted.  Usage is
        recorded in the ledger on miss.
        """
        # ── Compute keys and classify hits / misses ──────────────────
        keys: list[str] = []
        results: list[tuple[int, np.ndarray]] = []  # type: ignore[type-arg]
        missed_indices: list[int] = []
        missed_raw: list[str] = []

        for idx, raw_text in enumerate(texts):
            key = _embedding_cache_key(
                self.embedding_model_name,
                self._dimensions,
                raw_text,
                self._instruction_mode,
            )
            keys.append(key)
            cached = self._lookup(key)
            if cached is not None:
                results.append((idx, cached))
            else:
                missed_indices.append(idx)
                missed_raw.append(raw_text)

        # ── All cache hits → return immediately ──────────────────────
        if not missed_raw:
            results.sort(key=lambda item: item[0])
            return np.stack([v for _, v in results], axis=0).astype(np.float32)  # type: ignore[no-any-return]

        # ── API call for missed texts ────────────────────────────────
        # Deduplicate within the missed set so identical raw texts don't
        # hit the API twice in the same batch.
        unique_raw: list[str] = []
        raw_to_first_missed: dict[str, int] = {}
        for raw_text in missed_raw:
            if raw_text not in raw_to_first_missed:
                raw_to_first_missed[raw_text] = len(unique_raw)
                unique_raw.append(raw_text)

        preprocessed = _preprocess(unique_raw)

        # Budget reservation
        reservation_id = f"resv-{uuid.uuid4().hex[:12]}"
        upper_bound_tokens = sum(len(t) // 2 + 1 for t in preprocessed)
        upper_bound_usd = upper_bound_tokens * self._price_per_million / 1_000_000

        if self._ledger is not None:
            self._ledger.reserve(reservation_id, upper_bound_usd, self._project_limit_usd)

        try:
            started = time.perf_counter()
            vectors, actual_tokens = self._call_api(preprocessed)
            latency = time.perf_counter() - started
        except Exception:
            if self._ledger is not None:
                self._ledger.release(reservation_id, "invoke-failed")
            raise

        # ── Store and settle ─────────────────────────────────────────
        if self._ledger is not None:
            events: list[LedgerEntry] = []
            per_text_tokens = actual_tokens // len(unique_raw)
            remainder = actual_tokens % len(unique_raw)
            for ui, raw_text in enumerate(unique_raw):
                tok = per_text_tokens + (1 if ui < remainder else 0)
                key = _embedding_cache_key(
                    self.embedding_model_name,
                    self._dimensions,
                    raw_text,
                    self._instruction_mode,
                )
                event_id = f"emb-{key[:16]}"
                entry = LedgerEntry(
                    event_id=event_id,
                    reservation_id=reservation_id,
                    stage="embedding",
                    model=self.embedding_model_name,
                    cache_hit=False,
                    batch_discount_applied=False,
                    actual_usd=max(0.0, tok * self._price_per_million / 1_000_000),
                    method_equivalent_usd=max(
                        0.0, tok * self._price_per_million / 1_000_000
                    ),
                    embedding_tokens=tok,
                    observed_latency_seconds=latency / len(unique_raw),
                    method_equivalent_latency_seconds=latency / len(unique_raw),
                )
                events.append(entry)

            self._ledger.settle(reservation_id, events)

        # ── Store vectors in cache ───────────────────────────────────
        for _ui, (raw_text, vector) in enumerate(zip(unique_raw, vectors, strict=True)):
            key = _embedding_cache_key(
                self.embedding_model_name,
                self._dimensions,
                raw_text,
                self._instruction_mode,
            )
            self._store(key, raw_text, vector)

        # ── Map missed indices back to vectors ───────────────────────
        unique_vec_map = {raw: vec for raw, vec in zip(unique_raw, vectors, strict=True)}

        for raw_text, orig_idx in zip(missed_raw, missed_indices, strict=True):
            vector = unique_vec_map[raw_text]
            results.append((orig_idx, vector))

        results.sort(key=lambda item: item[0])
        return np.stack([v for _, v in results], axis=0).astype(np.float32)  # type: ignore[no-any-return]

    # ── batch_encode (chunked, normalised) ─────────────────────────────

    def batch_encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:  # type: ignore[type-arg]
        """Encode *texts* in batches of at most 64, then optionally L2-normalise.

        Mirrors the upstream contract: chunks are processed via ``encode()``,
        concatenated in input order, and row-wise normalised when
        ``embedding_config.norm`` is true.  Never uses the upstream bare
        ``except`` / ``ipdb`` path.
        """
        if isinstance(texts, str):
            texts = [texts]

        from copy import deepcopy

        params: dict[str, Any] = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        # Upstream instruction handling — accept but deliberately do not
        # prepend in ``upstream_ignored`` mode (the instruction is part of
        # the cache key already).
        if "instruction" in kwargs and kwargs["instruction"] != "":
            params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "

        batch_size = int(params.pop("batch_size", 16))
        # Clamp to the 64-item ceiling required by the plan.
        batch_size = min(batch_size, 64)

        chunks: list[np.ndarray] = []  # type: ignore[type-arg]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Our encode() handles caching — no need for upstream's bare except/ipdb.
            chunks.append(self.encode(batch))

        results = np.concatenate(chunks, axis=0)

        # Convert from torch.Tensor if needed (upstream compatibility).
        import torch

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()

        if self.embedding_config.norm:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            # Protect against zero vectors
            norms = np.where(norms == 0, 1.0, norms)
            results = results / norms

        return results.astype(np.float32)  # type: ignore[no-any-return]

    # ── Upstream get_query_doc_scores (pass-through) ─────────────────────

    def get_query_doc_scores(
        self, query_vec: np.ndarray, doc_vecs: np.ndarray  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Dot-product similarity (delegates to upstream implementation)."""
        return np.dot(query_vec, doc_vecs.T)  # type: ignore[no-any-return]


# ── Engine injection ─────────────────────────────────────────────────────────


def inject_embedding_model(engine: Any, model: Any) -> None:
    """Install *model* as the embedding backend for *engine* and its three stores.

    Must be called after HippoRAG construction and **before** ``index()`` or
    ``retrieve()``.  The engine and its stores keep a reference to the same
    model instance so that the persistent cache is shared.
    """
    engine.embedding_model = model
    engine.chunk_embedding_store.embedding_model = model
    engine.entity_embedding_store.embedding_model = model
    engine.fact_embedding_store.embedding_model = model


# ── Query-embedding export ───────────────────────────────────────────────────


def export_query_embeddings(engine: Any, path: Path) -> None:
    """Persist ``engine.query_to_embedding`` as a compressed NumPy archive.

    The archive is keyed by query string and can be reloaded across runs when
    the dataset, corpus SHA, model, dimensions, instruction mode, upstream SHA,
    patch SHA, and index-config hash all match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}  # type: ignore[type-arg]
    for query, embedding in engine.query_to_embedding.items():
        arr = np.asarray(embedding, dtype=np.float32)
        # Ensure at least 1-D
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays[query] = arr
    np.savez_compressed(str(path), **arrays)
