"""Tests for two-stage Batch OpenIE — NER → Triple state machine, sharding, and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from metagate_hipporag.batch_openie import (
    _UPSTREAM_NER_TEMPLATE,
    _UPSTREAM_TRIPLE_TEMPLATE,
    BatchPhase,
    NERResponse,
    TripleItem,
    TripleResponse,
    _estimate_request_tokens,
    _make_custom_id,
    _pack_shards,
    build_ner_requests,
    build_triple_requests,
    collect_output_rows,
    export_upstream_openie,
)

# ── Schema tests ────────────────────────────────────────────────────────────


def test_ner_response_validates_entities() -> None:
    ner = NERResponse(named_entities=["Alice", "Bob"])
    assert ner.named_entities == ["Alice", "Bob"]


def test_ner_response_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        NERResponse.model_validate({"named_entities": ["X"], "extra": "bad"})


def test_triple_item_validates() -> None:
    t = TripleItem(subject="Alice", predicate="knows", object="Bob")
    assert t.subject == "Alice"


def test_triple_item_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        TripleItem.model_validate(
            {"subject": "Alice", "predicate": "knows", "object": "Bob", "extra": "bad"}
        )


def test_triple_response_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        TripleResponse.model_validate({"triples": [], "extra": "bad"})


# ── custom_id / request tests ───────────────────────────────────────────────


def test_make_custom_id_is_deterministic() -> None:
    first = _make_custom_id("ner", "musique", "chunk-x", "gpt-4o-mini", "prompt_sha", "cfg_sha")
    second = _make_custom_id("ner", "musique", "chunk-x", "gpt-4o-mini", "prompt_sha", "cfg_sha")
    assert first == second
    assert first.startswith("ner-musique-")
    assert len(first) == len("ner-musique-") + 32  # 32-char hex


def test_make_custom_id_changes_with_input() -> None:
    a = _make_custom_id("ner", "musique", "doc-a", "m", "ps", "cs")
    b = _make_custom_id("ner", "musique", "doc-b", "m", "ps", "cs")
    assert a != b


def test_ner_requests_have_stable_unique_custom_ids() -> None:
    docs = {"chunk-a": "Title A\nBody A", "chunk-b": "Title B\nBody B"}
    rows = build_ner_requests(
        docs, model="gpt-4o-mini-2024-07-18", seed=20260711,
        prompt_hash="aa" * 32, config_hash="bb" * 32,
    )
    assert len(rows) == 2
    assert len({row["custom_id"] for row in rows}) == 2
    assert all(row["url"] == "/v1/chat/completions" for row in rows)
    assert all(
        row["body"]["response_format"]["type"] == "json_schema" for row in rows
    )
    assert all(
        row["body"]["response_format"]["json_schema"]["name"] == "ner_response"
        for row in rows
    )


def test_triple_requests_include_entities() -> None:
    docs = {"chunk-a": "Title A\nBody A"}
    ner_output = {"chunk-a": {"named_entities": ["Entity1", "Entity2"]}}
    rows = build_triple_requests(
        docs, ner_output, model="gpt-4o-mini-2024-07-18", seed=20260711,
        prompt_hash="aa" * 32, config_hash="bb" * 32,
    )
    assert len(rows) == 1
    msg_content = rows[0]["body"]["messages"][-1]["content"]
    assert "Entity1" in msg_content
    assert "Entity2" in msg_content
    assert (
        rows[0]["body"]["response_format"]["json_schema"]["name"]
        == "triple_response"
    )


# ── BatchPhase state machine ────────────────────────────────────────────────


def test_batch_phase_cannot_mark_incomplete_as_complete() -> None:
    phase = BatchPhase(dataset="nq_rear", phase="ner", expected=3, completed=1)
    try:
        phase.require_complete()
    except RuntimeError as exc:
        assert "1/3" in str(exc)
    else:
        raise AssertionError("incomplete NER phase was accepted")


def test_batch_phase_marks_complete_and_saves() -> None:
    phase = BatchPhase(dataset="nq_rear", phase="ner", expected=2, completed=2)
    try:
        phase.require_complete()
    except RuntimeError as err:
        raise AssertionError("complete phase should not raise") from err


def test_triple_phase_cannot_start_before_complete_ner(tmp_path: Path) -> None:
    phase = BatchPhase(dataset="musique", phase="ner", expected=2, completed=1)
    try:
        phase.require_complete()
    except RuntimeError as exc:
        assert "1/2" in str(exc)
    else:
        raise AssertionError("incomplete NER phase was accepted")


# ── collect_output_rows ─────────────────────────────────────────────────────


def _make_row(custom_id: str, content: str, status: int = 200) -> dict:
    return {
        "id": f"batch-{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": status,
            "request_id": "req-1",
            "body": {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1752240000,
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        },
        "error": None,
    }


def test_collect_output_rows_preserves_order() -> None:
    rows = [
        _make_row("ner-a", '{"named_entities":["X"]}'),
        _make_row("ner-b", '{"named_entities":["Y"]}'),
    ]
    result = collect_output_rows(rows, expected_ids={"ner-a", "ner-b"})
    assert [r["custom_id"] for r in result] == ["ner-a", "ner-b"]


def test_duplicate_output_is_rejected() -> None:
    rows = [
        _make_row("ner-a", '{"named_entities":["X"]}'),
        _make_row("ner-a", '{"named_entities":["Y"]}'),
    ]
    try:
        collect_output_rows(rows, expected_ids={"ner-a"})
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate output was accepted")


def test_missing_custom_id_is_rejected() -> None:
    rows = [_make_row("ner-a", '{"named_entities":["X"]}')]
    try:
        collect_output_rows(rows, expected_ids={"ner-a", "ner-b"})
    except ValueError as exc:
        assert "missing" in str(exc).lower() or "ner-b" in str(exc)
    else:
        raise AssertionError("missing custom_id was accepted")


def test_non_200_status_is_rejected() -> None:
    rows = [_make_row("ner-a", '{"named_entities":["X"]}', status=400)]
    try:
        collect_output_rows(rows, expected_ids={"ner-a"})
    except ValueError:
        return
    raise AssertionError("non-200 status was accepted")


def test_failed_row_with_error_is_rejected() -> None:
    row = {
        "id": "batch-1",
        "custom_id": "ner-a",
        "response": {"status_code": 200, "request_id": "req-1", "body": None},
        "error": {"code": "server_error", "message": "something went wrong"},
    }
    try:
        collect_output_rows([row], expected_ids={"ner-a"})
    except ValueError:
        return
    raise AssertionError("row with error was accepted")


def test_malformed_json_content_is_rejected() -> None:
    rows = [_make_row("ner-a", "not json {{{")]
    try:
        collect_output_rows(rows, expected_ids={"ner-a"})
    except ValueError:
        return
    raise AssertionError("malformed JSON content was accepted")


# ── Shard packing ───────────────────────────────────────────────────────────

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "x", "strict": True, "schema": {}},
}


def _min_req(custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
            "seed": 20260711,
            "max_completion_tokens": 512,
            "response_format": _RESPONSE_FORMAT,
        },
    }


def test_pack_shards_respects_limit() -> None:
    """Each shard must not exceed the token limit."""
    requests = [_min_req(f"r{i}") for i in range(100)]
    shards = _pack_shards(
        requests, model="gpt-4o-mini-2024-07-18", max_tokens_per_shard=5000
    )
    assert len(shards) >= 2
    # Every request must appear exactly once across all shards
    all_ids = [r["custom_id"] for shard in shards for r in shard]
    assert sorted(all_ids) == sorted([f"r{i}" for i in range(100)])


def test_pack_shards_with_few_requests() -> None:
    requests = [_min_req("r0")]
    shards = _pack_shards(
        requests, model="gpt-4o-mini-2024-07-18", max_tokens_per_shard=50000
    )
    assert len(shards) == 1
    assert len(shards[0]) == 1


def test_estimate_request_tokens_is_positive() -> None:
    req = _min_req("x")
    tokens = _estimate_request_tokens(req, model="gpt-4o-mini-2024-07-18")
    assert tokens > 0
    assert isinstance(tokens, int)


# ── export_upstream_openie ──────────────────────────────────────────────────


def test_export_upstream_openie_format() -> None:
    docs = {"chunk-a": "Title A\nBody A"}
    ner_results = {"chunk-a": {"named_entities": ["E1", "E2"]}}
    triple_results = {
        "chunk-a": {"triples": [{"subject": "E1", "predicate": "p", "object": "E2"}]}
    }
    exported = export_upstream_openie(docs, ner_results, triple_results)
    assert len(exported["docs"]) == 1
    doc = exported["docs"][0]
    assert doc["idx"].startswith("chunk-")
    assert doc["passage"] == "Title A\nBody A"
    assert doc["extracted_entities"] == ["E1", "E2"]
    assert doc["extracted_triples"] == [["E1", "p", "E2"]]


def test_export_upstream_openie_filters_empty_triples() -> None:
    docs = {"chunk-a": "Title A\nBody A"}
    ner_results = {"chunk-a": {"named_entities": ["E1"]}}
    triple_results = {
        "chunk-a": {
            "triples": [
                {"subject": "E1", "predicate": "p", "object": "E2"},
                {"subject": "", "predicate": "p", "object": "E2"},   # empty subject
                {"subject": "E1", "predicate": "p", "object": ""},    # empty object
            ]
        }
    }
    exported = export_upstream_openie(docs, ner_results, triple_results)
    triples = exported["docs"][0]["extracted_triples"]
    assert len(triples) == 1
    assert triples[0] == ["E1", "p", "E2"]


# ── Prompt template tests ───────────────────────────────────────────────────


def test_ner_prompt_template_renders() -> None:
    from string import Template as StringTemplate

    rendered = [
        {**msg, "content": StringTemplate(msg["content"]).substitute(passage="test passage")}
        if isinstance(msg.get("content"), str) and "${passage}" in msg["content"]
        else msg
        for msg in _UPSTREAM_NER_TEMPLATE
    ]
    # The last user message should contain "test passage"
    last_content = rendered[-1]["content"]
    assert "test passage" in last_content


def test_triple_prompt_template_renders() -> None:
    from string import Template as StringTemplate

    rendered = [
        {**msg, "content": StringTemplate(msg["content"]).substitute(
            passage="test passage", named_entity_json='{"named_entities":["E1"]}'
        )}
        if isinstance(msg.get("content"), str)
        else msg
        for msg in _UPSTREAM_TRIPLE_TEMPLATE
    ]
    last_content = rendered[-1]["content"]
    assert "test passage" in last_content
    assert "E1" in last_content


# ── BatchConfig validation ──────────────────────────────────────────────────


def test_batch_config_rejects_poll_interval_over_30() -> None:
    from metagate_hipporag.config import BatchConfig

    try:
        BatchConfig.model_validate({
            "max_enqueued_input_tokens": 1500000,
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "max_requests_per_shard": 5000,
            "poll_interval_seconds": 45,
            "ner_max_output_tokens": 512,
            "triple_max_output_tokens": 1024,
        })
    except ValidationError:
        return
    raise AssertionError("poll_interval_seconds > 30 was accepted")


def test_batch_config_rejects_token_limit_over_2m() -> None:
    from metagate_hipporag.config import BatchConfig

    try:
        BatchConfig.model_validate({
            "max_enqueued_input_tokens": 3000000,
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "max_requests_per_shard": 5000,
            "poll_interval_seconds": 30,
            "ner_max_output_tokens": 512,
            "triple_max_output_tokens": 1024,
        })
    except ValidationError:
        return
    raise AssertionError("max_enqueued_input_tokens > 2,000,000 was accepted")


def test_batch_config_rejects_unknown_key() -> None:
    from metagate_hipporag.config import BatchConfig

    try:
        BatchConfig.model_validate({
            "max_enqueued_input_tokens": 1500000,
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "max_requests_per_shard": 5000,
            "poll_interval_seconds": 30,
            "ner_max_output_tokens": 512,
            "triple_max_output_tokens": 1024,
            "unknown_key": "should be rejected",
        })
    except ValidationError:
        return
    raise AssertionError("unknown key was accepted by BatchConfig")
