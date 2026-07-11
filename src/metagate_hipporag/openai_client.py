"""Cached, structured OpenAI client with SQLite persistence, retry, and cost accounting.

Cache key is a SHA-256 digest of the canonical API payload (model, messages,
schema, parameters).  The ``OPENAI_API_KEY`` and ``Authorization`` header are
*never* serialized into the cache key.

Every cache miss reserves budget in the ``UsageLedger`` (when provided), calls
the real or fake backend, and atomically settles a ``LedgerEntry``.  Cache hits
record a zero-actual-cost consumption event so method-equivalent accounting
stays accurate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import tiktoken
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import LedgerEntry, Usage
from .provenance import UsageLedger

T = TypeVar("T", bound=BaseModel)

# ── Data --------------------------------------------------------------------


@dataclass(frozen=True)
class RawCompletion:
    """The raw bytes the backend returned, before Pydantic validation."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class StructuredCompletion(Generic[T]):
    """A validated model instance paired with its token / cost usage."""

    value: T
    usage: Usage


# ── Cache key ---------------------------------------------------------------

# Fields that carry secrets and must be excluded from the canonical payload.
_SECRET_FIELDS = frozenset({"api_key", "Authorization", "authorization"})


def _canonical_payload(payload: dict[str, Any]) -> str:
    """Serialize *payload* deterministically, stripping secret fields."""
    safe = {k: v for k, v in payload.items() if k not in _SECRET_FIELDS}
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cache_key(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical (non-secret) payload."""
    return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()


# ── Token estimation --------------------------------------------------------

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for *model*, falling back to cl100k_base."""
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model]


def _estimate_input_tokens(
    model: str, messages: list[dict[str, Any]], schema: dict[str, Any]
) -> int:
    """Conservative estimate of prompt tokens (messages + serialized JSON schema).

    The estimate errs on the high side so the budget reservation is safe.
    """
    enc = _get_encoding(model)
    # Messages contribute N tokens per tokenized string.
    msg_tokens = sum(
        len(enc.encode(str(msg.get("content", "")))) for msg in messages
    )
    # Schema JSON contributes roughly its serialized length in tokens.
    schema_str = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    schema_tokens = len(enc.encode(schema_str))
    # Add a 20 % padding for the system prompt / role tokens / formatting.
    return int((msg_tokens + schema_tokens) * 1.2)


# ── Client ------------------------------------------------------------------


def _default_invoke_factory() -> Callable[..., RawCompletion]:
    """Build the real OpenAI backend (called lazily so tests can inject fakes)."""
    return _invoke_openai


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError)
    ),
    reraise=True,
)
def _invoke_openai(**kwargs: Any) -> RawCompletion:
    """Call the OpenAI Chat Completions API with structured output.

    *Not* a staticmethod so that tenacity can wrap it cleanly.
    Authentication, permission, bad-request, and schema errors are NOT
    retried — they fail immediately (tenacity's reraise lets them through).
    """
    client = OpenAI()
    started = time.perf_counter()

    response = client.chat.completions.create(
        model=kwargs["model"],
        messages=kwargs["messages"],
        max_completion_tokens=kwargs["max_completion_tokens"],
        seed=kwargs["seed"],
        temperature=kwargs["temperature"],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": kwargs["response_model"].__name__.lower(),
                "strict": True,
                "schema": kwargs["response_model"].model_json_schema(),
            },
        },
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI returned empty content")
    if response.usage is None:
        raise ValueError("OpenAI returned no usage object")
    return RawCompletion(
        content=content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        latency_seconds=time.perf_counter() - started,
    )


class CachedStructuredClient:
    """Structured OpenAI client with SQLite cache, retry, and usage ledger."""

    def __init__(
        self,
        cache_path: Path,
        *,
        invoke: Callable[..., RawCompletion] | None = None,
        input_price_per_million: float = 0.15,
        output_price_per_million: float = 0.60,
        ledger: UsageLedger | None = None,
        project_limit_usd: float = 18.0,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self._invoke_impl = invoke or _invoke_openai
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.ledger = ledger
        self.project_limit_usd = project_limit_usd

        if invoke is None and ledger is None:
            raise ValueError(
                "a production OpenAI client requires a UsageLedger; "
                "pass ledger=UsageLedger(...) or inject a fake invoke for testing"
            )

        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS completions ("
                "key TEXT PRIMARY KEY, "
                "custom_id TEXT NOT NULL, "
                "response TEXT NOT NULL, "
                "usage TEXT NOT NULL"
                ")"
            )
            conn.commit()

    # ── Public API ──────────────────────────────────────────────────────

    def complete(
        self,
        *,
        custom_id: str,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_completion_tokens: int,
        seed: int,
        temperature: float,
    ) -> StructuredCompletion[T]:
        """Return a validated structured completion, served from cache when possible.

        On a cache **hit**: usage is returned with ``cache_hit=True`` and zero
        actual cost; method-equivalent cost is preserved from the original call.

        On a cache **miss**: budget is reserved in the ledger (if available),
        the backend is invoked, and the result is cached.  The ledger entry is
        settled atomically.  On pre-submission failure the reservation is
        released.
        """
        started = time.perf_counter()

        schema = response_model.model_json_schema()
        payload: dict[str, Any] = {
            "base_url": "https://api.openai.com/v1",
            "model": model,
            "messages": messages,
            "schema": schema,
            "max_completion_tokens": max_completion_tokens,
            "seed": seed,
            "temperature": temperature,
        }
        key = _cache_key(payload)

        # ── Cache hit path ──────────────────────────────────────────────
        with sqlite3.connect(str(self.cache_path)) as conn:
            row = conn.execute(
                "SELECT response, usage FROM completions WHERE key = ?", (key,)
            ).fetchone()

        if row is not None:
            cached_usage = Usage.model_validate_json(row[1]).model_copy(
                update={
                    "cache_hit": True,
                    "actual_usd": 0.0,
                    "observed_latency_seconds": time.perf_counter() - started,
                }
            )
            value = response_model.model_validate_json(row[0])
            # Record an idempotent zero-cost consumption event for accounting.
            if self.ledger is not None:
                event_id = f"cache-{custom_id}-{key[:12]}"
                entry = LedgerEntry(
                    event_id=event_id,
                    reservation_id="cache-hit",
                    stage="retrieval_llm",  # closest match; caller context is lost here
                    model=model,
                    cache_hit=True,
                    batch_discount_applied=False,
                    actual_usd=0.0,
                    method_equivalent_usd=cached_usage.method_equivalent_usd,
                    prompt_tokens=0,
                    completion_tokens=0,
                    observed_latency_seconds=time.perf_counter() - started,
                    method_equivalent_latency_seconds=cached_usage.method_equivalent_latency_seconds,
                )
                self.ledger.append(entry)

            return StructuredCompletion(value=value, usage=cached_usage)

        # ── Cache miss path ─────────────────────────────────────────────
        # Estimate upper-bound cost and reserve budget.
        estimated_input = _estimate_input_tokens(model, messages, schema)
        upper_bound = (
            estimated_input * self.input_price
            + max_completion_tokens * self.output_price
        ) / 1_000_000

        reservation_id = f"resv-{uuid.uuid4().hex[:12]}"
        if self.ledger is not None:
            self.ledger.reserve(reservation_id, upper_bound, self.project_limit_usd)

        try:
            raw = self._invoke_impl(
                custom_id=custom_id,
                model=model,
                messages=messages,
                response_model=response_model,
                max_completion_tokens=max_completion_tokens,
                seed=seed,
                temperature=temperature,
            )
        except Exception:
            # Pre-submission or transient error after retries exhausted → release.
            if self.ledger is not None:
                self.ledger.release(reservation_id, "invoke-failed")
            raise

        # Validate the structured output.
        value = response_model.model_validate_json(raw.content)
        actual_cost = (
            raw.prompt_tokens * self.input_price
            + raw.completion_tokens * self.output_price
        ) / 1_000_000

        usage = Usage(
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            observed_latency_seconds=raw.latency_seconds,
            method_equivalent_latency_seconds=raw.latency_seconds,
            cache_hit=False,
            actual_usd=actual_cost,
            method_equivalent_usd=actual_cost,
        )

        # Persist in cache.
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.execute(
                "INSERT INTO completions(key, custom_id, response, usage) "
                "VALUES (?, ?, ?, ?)",
                (key, custom_id, value.model_dump_json(), usage.model_dump_json()),
            )
            conn.commit()

        # Settle the ledger reservation.
        if self.ledger is not None:
            event_id = f"llm-{custom_id}-{key[:12]}"
            entry = LedgerEntry(
                event_id=event_id,
                reservation_id=reservation_id,
                stage="retrieval_llm",
                model=model,
                cache_hit=False,
                batch_discount_applied=False,
                actual_usd=actual_cost,
                method_equivalent_usd=actual_cost,
                prompt_tokens=raw.prompt_tokens,
                completion_tokens=raw.completion_tokens,
                observed_latency_seconds=raw.latency_seconds,
                method_equivalent_latency_seconds=raw.latency_seconds,
            )
            self.ledger.settle(reservation_id, [entry])

        return StructuredCompletion(value=value, usage=usage)
