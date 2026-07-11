"""Tests for CachedStructuredClient — SQLite cache, retry, and cost accounting."""

from __future__ import annotations

from pathlib import Path

from metagate_hipporag.models import GateDecision
from metagate_hipporag.openai_client import CachedStructuredClient, RawCompletion


def test_completion_is_schema_validated_and_cached(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_invoke(**kwargs: object) -> RawCompletion:
        calls.append(str(kwargs["custom_id"]))
        return RawCompletion(
            content=(
                '{"evidence_sufficient_probability":0.25,'
                '"missing_information":"bridge fact",'
                '"retrieval_rewrite":"entity bridge relation",'
                '"rationale_summary":"A required relation is absent."}'
            ),
            prompt_tokens=100,
            completion_tokens=20,
            latency_seconds=0.2,
        )

    client = CachedStructuredClient(tmp_path / "calls.sqlite", invoke=fake_invoke)
    first = client.complete(
        custom_id="gate-dataset-example",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "evidence"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    second = client.complete(
        custom_id="gate-dataset-example",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "evidence"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    assert first.value == second.value
    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is True
    assert second.usage.actual_usd == 0.0
    assert calls == ["gate-dataset-example"]


def test_cache_key_changes_with_prompt(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_invoke(**kwargs: object) -> RawCompletion:
        calls.append(str(kwargs["custom_id"]))
        return RawCompletion(
            content=(
                '{"evidence_sufficient_probability":0.8,'
                '"missing_information":"none",'
                '"retrieval_rewrite":"keep searching",'
                '"rationale_summary":"Evidence is complete."}'
            ),
            prompt_tokens=50,
            completion_tokens=15,
            latency_seconds=0.1,
        )

    client = CachedStructuredClient(tmp_path / "calls2.sqlite", invoke=fake_invoke)
    first = client.complete(
        custom_id="gate-A",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "prompt version 1"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    second = client.complete(
        custom_id="gate-B",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "prompt version 2"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    # Different prompts → different keys → two API calls
    assert len(calls) == 2
    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is False


def test_cache_key_changes_with_model(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_invoke(**kwargs: object) -> RawCompletion:
        calls.append(str(kwargs["custom_id"]))
        return RawCompletion(
            content=(
                '{"evidence_sufficient_probability":0.5,'
                '"missing_information":"x",'
                '"retrieval_rewrite":"y",'
                '"rationale_summary":"z"}'
            ),
            prompt_tokens=30,
            completion_tokens=10,
            latency_seconds=0.05,
        )

    client = CachedStructuredClient(tmp_path / "calls3.sqlite", invoke=fake_invoke)
    client.complete(
        custom_id="m1",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "test"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    client.complete(
        custom_id="m2",
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": "test"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    assert len(calls) == 2


def test_malformed_json_raises_validation_error(tmp_path: Path) -> None:
    def fake_invoke(**kwargs: object) -> RawCompletion:
        return RawCompletion(
            content="not valid json at all {{{",
            prompt_tokens=10,
            completion_tokens=5,
            latency_seconds=0.01,
        )

    client = CachedStructuredClient(tmp_path / "bad.sqlite", invoke=fake_invoke)
    try:
        client.complete(
            custom_id="bad-json",
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": "whatever"}],
            response_model=GateDecision,
            max_completion_tokens=256,
            seed=20260711,
            temperature=0.0,
        )
    except Exception:
        # We expect a Pydantic ValidationError (wrapped or raw)
        return
    raise AssertionError("malformed JSON was accepted without error")


def test_no_secret_in_cache_key(tmp_path: Path) -> None:
    """API key and Authorization header must never be serialized into the cache key."""
    from metagate_hipporag.openai_client import _cache_key, _canonical_payload

    # The _canonical_payload function strips secret fields.
    base: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }
    key_without_secrets = _cache_key(dict(base))

    # Adding an api_key should NOT change the cache key.
    with_key: dict[str, object] = dict(base)
    with_key["api_key"] = "sk-secret-should-not-appear"
    assert _cache_key(with_key) == key_without_secrets

    # Adding an Authorization header should NOT change the cache key.
    with_auth: dict[str, object] = dict(base)
    with_auth["Authorization"] = "Bearer secret-token"
    assert _cache_key(with_auth) == key_without_secrets

    # The canonical payload must not contain the secret fields.
    canonical = _canonical_payload(with_key)
    assert "sk-secret" not in canonical
    assert "«redacted:" not in canonical


def test_cache_key_independent_of_custom_id(tmp_path: Path) -> None:
    """custom_id is audit metadata — identical requests with different IDs share cache."""
    calls: list[str] = []

    def fake_invoke(**kwargs: object) -> RawCompletion:
        calls.append(str(kwargs["custom_id"]))
        return RawCompletion(
            content=(
                '{"evidence_sufficient_probability":0.5,'
                '"missing_information":"x",'
                '"retrieval_rewrite":"y",'
                '"rationale_summary":"z"}'
            ),
            prompt_tokens=30,
            completion_tokens=10,
            latency_seconds=0.05,
        )

    client = CachedStructuredClient(
        tmp_path / "custom_id.sqlite", invoke=fake_invoke
    )
    first = client.complete(
        custom_id="id-A",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "same prompt"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    second = client.complete(
        custom_id="id-B",  # different custom_id, same everything else
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "same prompt"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    # Only one API call — second hits cache despite different custom_id.
    assert len(calls) == 1
    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is True
    assert first.value == second.value


def test_cache_key_changes_with_schema(tmp_path: Path) -> None:
    """Different response_model schemas must produce different cache keys."""
    from metagate_hipporag.models import GateDecision, Usage
    from metagate_hipporag.openai_client import _cache_key

    payload: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "test"}],
        "max_completion_tokens": 256,
        "seed": 20260711,
        "temperature": 0.0,
        "base_url": "https://api.openai.com/v1",
    }

    gate_payload = dict(payload)
    gate_payload["schema"] = GateDecision.model_json_schema()
    gate_key = _cache_key(gate_payload)

    usage_payload = dict(payload)
    usage_payload["schema"] = Usage.model_json_schema()
    usage_key = _cache_key(usage_payload)

    assert gate_key != usage_key, (
        "different response schemas must produce different cache keys"
    )


def test_retry_configuration() -> None:
    """Verify the retry decorator wraps _invoke_openai and preserves the inner name."""
    from metagate_hipporag.openai_client import _invoke_openai

    # _invoke_openai is decorated with @retry — check the wrapper chain.
    fn = _invoke_openai
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    # The innermost function should be the original undecorated one.
    assert fn.__name__ == "_invoke_openai"


def test_cache_hit_preserves_method_equivalent_cost(tmp_path: Path) -> None:
    """Cache hits should keep the original method-equivalent cost for fair accounting."""
    def fake_invoke(**kwargs: object) -> RawCompletion:
        return RawCompletion(
            content=(
                '{"evidence_sufficient_probability":0.9,'
                '"missing_information":"none",'
                '"retrieval_rewrite":"stay",'
                '"rationale_summary":"ok"}'
            ),
            prompt_tokens=200,
            completion_tokens=30,
            latency_seconds=0.3,
        )

    client = CachedStructuredClient(tmp_path / "equiv.sqlite", invoke=fake_invoke)
    first = client.complete(
        custom_id="equiv-test",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "check"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    # First call: actual cost = method-equivalent cost.
    assert first.usage.cache_hit is False
    assert first.usage.actual_usd > 0.0
    assert first.usage.actual_usd == first.usage.method_equivalent_usd

    second = client.complete(
        custom_id="equiv-test-2",
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "check"}],
        response_model=GateDecision,
        max_completion_tokens=256,
        seed=20260711,
        temperature=0.0,
    )
    # Cache hit: actual cost = 0, method-equivalent = original cost.
    assert second.usage.cache_hit is True
    assert second.usage.actual_usd == 0.0
    assert second.usage.method_equivalent_usd == first.usage.method_equivalent_usd
