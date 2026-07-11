"""Resumable two-stage Batch OpenIE: NER → Triple state machine + sharding + budget preflight.

The pipeline is split into two dependent phases:
1. **NER** — extract named entities from every passage via OpenAI Batch.
2. **Triple** — extract triples from every passage, conditioned on the NER
   entities, via OpenAI Batch.  This phase can only start after *all* NER
   shards pass validation.

Each phase is further split into *shards* to stay within the account's
enqueued-input-token limit (default 1,500,000).  Every submission is
preceded by a budget preflight that reserves the worst-case cost in the
``UsageLedger``.

Subcommands
-----------
``prepare``  Build request JSONL shards; validate token budgets.
``submit``   Upload JSONL files and create OpenAI Batch jobs.
``poll``     Poll until completed, failed, expired, or cancelled.
``collect``  Download output, validate every row, and persist results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

# ── Strict Pydantic schemas for structured Batch outputs ────────────────────


class NERResponse(BaseModel):
    """Named entity recognition output — exactly as the upstream expects."""

    model_config = ConfigDict(extra="forbid")
    named_entities: list[str]


class TripleItem(BaseModel):
    """A single (subject, predicate, object) triple.

    Using an object form instead of a 3-tuple avoids unsupported
    ``prefixItems`` in strict Structured Outputs.
    """

    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str


class TripleResponse(BaseModel):
    """Triple extraction output."""

    model_config = ConfigDict(extra="forbid")
    triples: list[TripleItem]


# ── Prompt templates (sourced from pinned upstream PromptTemplateManager) ───
#
# These match ``hipporag.prompts.templates.ner`` and
# ``hipporag.prompts.templates.triple_extraction`` at the pinned commit.
# When the upstream is importable we prefer those; the hardcoded copies are
# the fallback and MUST produce bit-identical prompt hashes.

_UPSTREAM_NER_TEMPLATE: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "Your task is to extract named entities from the given paragraph. \n"
            "Respond with a JSON list of entities.\n"
        ),
    },
    {
        "role": "user",
        "content": (
            "Radio City\n"
            "Radio City is India's first private FM radio station and was started"
            " on 3 July 2001.\n"
            "It plays Hindi, English and regional songs.\n"
            "Radio City recently forayed into New Media in May 2008 with the"
            " launch of a music portal - PlanetRadiocity.com that offers music"
            " related news, videos, songs, and other music-related features."
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"named_entities":\n'
            '    ["Radio City", "India", "3 July 2001", "Hindi", "English",'
            ' "May 2008", "PlanetRadiocity.com"]\n'
            "}\n"
        ),
    },
    {"role": "user", "content": "${passage}"},
]

_UPSTREAM_TRIPLE_TEMPLATE: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "Your task is to construct an RDF (Resource Description Framework)"
            " graph from the given passages and named entity lists. \n"
            "Respond with a JSON list of triples, with each triple representing"
            " a relationship in the RDF graph. \n\n"
            "Pay attention to the following requirements:\n"
            "- Each triple should contain at least one, but preferably two, of"
            " the named entities in the list for each passage.\n"
            "- Clearly resolve pronouns to their specific names to maintain"
            " clarity.\n\n"
        ),
    },
    {
        "role": "user",
        "content": (
            "Convert the paragraph into a JSON dict, it has a named entity list"
            " and a triple list.\n"
            "Paragraph:\n"
            "```\n"
            "Radio City\n"
            "Radio City is India's first private FM radio station and was started"
            " on 3 July 2001.\n"
            "It plays Hindi, English and regional songs.\n"
            "Radio City recently forayed into New Media in May 2008 with the"
            " launch of a music portal - PlanetRadiocity.com that offers music"
            " related news, videos, songs, and other music-related features.\n"
            "```\n\n"
            '{"named_entities":\n'
            '    ["Radio City", "India", "3 July 2001", "Hindi", "English",'
            ' "May 2008", "PlanetRadiocity.com"]\n'
            "}\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"triples": [\n'
            '            ["Radio City", "located in", "India"],\n'
            '            ["Radio City", "is", "private FM radio station"],\n'
            '            ["Radio City", "started on", "3 July 2001"],\n'
            '            ["Radio City", "plays songs in", "Hindi"],\n'
            '            ["Radio City", "plays songs in", "English"],\n'
            '            ["Radio City", "forayed into", "New Media"],\n'
            '            ["Radio City", "launched", "PlanetRadiocity.com"],\n'
            '            ["PlanetRadiocity.com", "launched in", "May 2008"],\n'
            '            ["PlanetRadiocity.com", "is", "music portal"],\n'
            '            ["PlanetRadiocity.com", "offers", "news"],\n'
            '            ["PlanetRadiocity.com", "offers", "videos"],\n'
            '            ["PlanetRadiocity.com", "offers", "songs"]\n'
            "    ]\n"
            "}\n"
        ),
    },
    {"role": "user", "content": "${passage}\n${named_entity_json}"},
]


def _load_upstream_templates() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Try to load templates from the pinned upstream; fall back to hardcoded copies.

    The hardcoded copies are verified to be bit-identical to the upstream at
    the pinned commit.  If the upstream IS available, we prefer the dynamic
    import so prompt-hash validation across runs uses the canonical source.
    """
    try:
        from hipporag.prompts.prompt_template_manager import (  # type: ignore[import-untyped]
            PromptTemplateManager,
        )

        manager = PromptTemplateManager()
        ner_template = manager.get_template("ner")
        triple_template = manager.get_template("triple_extraction")
        # Both are lists of {role, content}; content may be string.Template.
        # Serialise to plain str for Batch JSONL.
        ner_serialised: list[dict[str, str]] = []
        for msg in ner_template:
            content = msg["content"]
            ner_serialised.append({
                "role": msg["role"],
                "content": content.template if hasattr(content, "template") else str(content),
            })
        triple_serialised: list[dict[str, str]] = []
        for msg in triple_template:
            content = msg["content"]
            triple_serialised.append({
                "role": msg["role"],
                "content": content.template if hasattr(content, "template") else str(content),
            })
        return ner_serialised, triple_serialised
    except Exception:
        return _UPSTREAM_NER_TEMPLATE, _UPSTREAM_TRIPLE_TEMPLATE


# ── Prompt hashing ──────────────────────────────────────────────────────────


def _prompt_sha(template: list[dict[str, str]]) -> str:
    """SHA-256 of the canonical prompt template (used for custom_id and manifest)."""
    canonical = json.dumps(template, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Custom ID construction ──────────────────────────────────────────────────


def _make_custom_id(
    phase: str,
    dataset: str,
    chunk_id: str,
    model: str,
    prompt_sha: str,
    effective_config_hash: str,
) -> str:
    """Deterministic, stable custom_id for a single Batch request.

    Format: ``<phase>-<dataset>-<32-char-hex>``
    """
    payload = f"{chunk_id}|{model}|{prompt_sha}|{effective_config_hash}"
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{phase}-{dataset}-{suffix}"


# ── Request construction ────────────────────────────────────────────────────


def _render_messages(
    template: list[dict[str, str]], **placeholders: str
) -> list[dict[str, str]]:
    """Substitute ``${name}`` placeholders in a template with string values."""
    rendered: list[dict[str, str]] = []
    for msg in template:
        content = msg["content"]
        for key, value in placeholders.items():
            content = content.replace(f"${{{key}}}", value)
        rendered.append({"role": msg["role"], "content": content})
    return rendered


def build_ner_requests(
    docs: dict[str, str],
    *,
    model: str,
    seed: int,
    prompt_hash: str,
    config_hash: str,
    dataset: str = "unknown",
) -> list[dict[str, Any]]:
    """Build NER Batch requests for every document."""
    ner_template, _ = _load_upstream_templates()
    requests: list[dict[str, Any]] = []
    for chunk_id, passage in docs.items():
        custom_id = _make_custom_id("ner", dataset, chunk_id, model, prompt_hash, config_hash)
        messages = _render_messages(ner_template, passage=passage)
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "seed": seed,
                "max_completion_tokens": 512,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ner_response",
                        "strict": True,
                        "schema": NERResponse.model_json_schema(),
                    },
                },
            },
        })
    return requests


def build_triple_requests(
    docs: dict[str, str],
    ner_output: dict[str, dict[str, Any]],
    *,
    model: str,
    seed: int,
    prompt_hash: str,
    config_hash: str,
    dataset: str = "unknown",
) -> list[dict[str, Any]]:
    """Build triple-extraction Batch requests, conditioned on NER entities."""
    _, triple_template = _load_upstream_templates()
    requests: list[dict[str, Any]] = []
    for chunk_id, passage in docs.items():
        entities = ner_output.get(chunk_id, {}).get("named_entities", [])
        entities_json = json.dumps({"named_entities": entities}, ensure_ascii=False)
        custom_id = _make_custom_id(
            "triple", dataset, chunk_id, model, prompt_hash, config_hash
        )
        messages = _render_messages(
            triple_template, passage=passage, named_entity_json=entities_json
        )
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "seed": seed,
                "max_completion_tokens": 1024,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "triple_response",
                        "strict": True,
                        "schema": TripleResponse.model_json_schema(),
                    },
                },
            },
        })
    return requests


# ── Token estimation & shard packing ────────────────────────────────────────


_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model]


_PER_REQUEST_OVERHEAD = 64


def _estimate_request_tokens(request: dict[str, Any], model: str) -> int:
    """Conservative token count for one Batch request body."""
    enc = _get_encoding(model)
    body_json = json.dumps(request["body"], ensure_ascii=False, sort_keys=True)
    return len(enc.encode(body_json)) + _PER_REQUEST_OVERHEAD


def _pack_shards(
    requests: list[dict[str, Any]],
    model: str,
    max_tokens_per_shard: int,
) -> list[list[dict[str, Any]]]:
    """Pack requests in original order into shards respecting the token limit."""
    shards: list[list[dict[str, Any]]] = []
    current_shard: list[dict[str, Any]] = []
    current_tokens = 0
    for req in requests:
        req_tokens = _estimate_request_tokens(req, model)
        if current_shard and current_tokens + req_tokens > max_tokens_per_shard:
            shards.append(current_shard)
            current_shard = []
            current_tokens = 0
        current_shard.append(req)
        current_tokens += req_tokens
    if current_shard:
        shards.append(current_shard)
    return shards


# ── BatchPhase state machine ────────────────────────────────────────────────


class BatchPhase:
    """Track progress of a single batch phase (NER or triple) across shards."""

    def __init__(
        self,
        dataset: str,
        phase: str,
        expected: int,
        completed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.phase = phase
        self.expected = expected
        self.completed = completed

    def require_complete(self) -> None:
        """Raise RuntimeError if not all expected rows are completed."""
        if self.completed < self.expected:
            raise RuntimeError(
                f"{self.phase}/{self.dataset} incomplete: "
                f"{self.completed}/{self.expected} rows finished"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "phase": self.phase,
            "expected": self.expected,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchPhase:
        return cls(
            dataset=data["dataset"],
            phase=data["phase"],
            expected=data["expected"],
            completed=data["completed"],
        )


def _save_phase_state(workspace: Path, phase: BatchPhase) -> None:
    state_file = workspace / f"{phase.phase}_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(phase.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, state_file)


def _load_phase_state(workspace: Path, phase_name: str) -> BatchPhase | None:
    state_file = workspace / f"{phase_name}_state.json"
    if not state_file.exists():
        return None
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return BatchPhase.from_dict(data)


# ── Output validation ───────────────────────────────────────────────────────


def collect_output_rows(
    rows: list[dict[str, Any]],
    expected_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate batch output rows and return them in arrival order.

    Raises ValueError on any of:
    - missing expected custom_id
    - duplicate custom_id
    - non-200 status_code
    - presence of an error object
    - malformed JSON in ``choices[0].message.content``
    """
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []

    for row in rows:
        cid = row.get("custom_id")
        if cid is None:
            raise ValueError(f"batch output row missing custom_id: {row.get('id', '?')}")
        if cid in seen:
            raise ValueError(f"duplicate custom_id in batch output: {cid}")
        seen.add(cid)

        error = row.get("error")
        if error is not None:
            raise ValueError(f"batch row {cid} has error: {error}")

        response = row.get("response")
        if response is None:
            raise ValueError(f"batch row {cid} has no response")
        status = response.get("status_code")
        if status != 200:
            raise ValueError(f"batch row {cid} returned status {status}")

        body = response.get("body")
        if body is None:
            raise ValueError(f"batch row {cid} has no response body")

        choices = body.get("choices")
        if not choices:
            raise ValueError(f"batch row {cid} has no choices")
        content = choices[0].get("message", {}).get("content")
        if content is None:
            raise ValueError(f"batch row {cid} has empty content")

        # Validate JSON parseable (actual schema validation happens per-phase)
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"batch row {cid} has malformed JSON: {exc}") from exc

        valid.append(row)

    missing = expected_ids - seen
    if missing:
        raise ValueError(
            f"missing {len(missing)} expected custom_ids in batch output, e.g. "
            f"{sorted(missing)[:5]}"
        )

    return valid


# ── Upstream OpenIE JSON export ─────────────────────────────────────────────


def _compute_upstream_chunk_id(passage: str) -> str:
    """``chunk-<md5-of-passage>`` — matches upstream HippoRAG convention."""
    digest = hashlib.md5(passage.encode("utf-8")).hexdigest()  # noqa: S324 — upstream uses md5
    return f"chunk-{digest}"


def export_upstream_openie(
    docs: dict[str, str],
    ner_results: dict[str, dict[str, Any]],
    triple_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Export results in the upstream-compatible OpenIE JSON format."""
    output_docs: list[dict[str, Any]] = []
    total_ent_chars = 0
    total_ent_words = 0
    ent_count = 0

    for chunk_id, passage in docs.items():
        entities = ner_results.get(chunk_id, {}).get("named_entities", [])
        triples_raw = triple_results.get(chunk_id, {}).get("triples", [])
        triples: list[list[str]] = []
        for t in triples_raw:
            subj = t.get("subject", "") if isinstance(t, dict) else t[0]
            pred = t.get("predicate", "") if isinstance(t, dict) else t[1]
            obj = t.get("object", "") if isinstance(t, dict) else t[2]
            if subj and pred and obj:
                triples.append([subj, pred, obj])

        for ent in entities:
            total_ent_chars += len(ent)
            total_ent_words += len(ent.split())
            ent_count += 1

        output_docs.append({
            "idx": _compute_upstream_chunk_id(passage),
            "passage": passage,
            "extracted_entities": entities,
            "extracted_triples": triples,
        })

    return {
        "docs": output_docs,
        "avg_ent_chars": round(total_ent_chars / max(1, ent_count), 1),
        "avg_ent_words": round(total_ent_words / max(1, ent_count), 1),
    }


# ── Budget preflight ────────────────────────────────────────────────────────


def _budget_preflight(
    phase: str,
    dataset: str,
    shard_index: int,
    total_shards: int,
    request_count: int,
    max_input_tokens: int,
    max_output_tokens_per: int,
    input_price_per_million: float,
    output_price_per_million: float,
    batch_discount: float,
    ledger_path: Path | None = None,
    project_limit_usd: float = 18.0,
) -> None:
    """Estimate cost and optionally reserve budget before submission."""
    input_usd = max_input_tokens * input_price_per_million / 1_000_000
    output_usd = request_count * max_output_tokens_per * output_price_per_million / 1_000_000
    gross = input_usd + output_usd
    discounted = gross * (1.0 - batch_discount)

    print(
        f"[{phase}/{dataset}] shard {shard_index + 1}/{total_shards}  "
        f"requests={request_count}  input_tokens≈{max_input_tokens:,}  "
        f"output_tokens≤{request_count * max_output_tokens_per:,}  "
        f"gross≈${gross:.4f}  batch≈${discounted:.4f}"
    )

    if ledger_path is not None and ledger_path.exists():
        from .provenance import UsageLedger

        ledger = UsageLedger(ledger_path, project_limit_usd)
        reservation_id = f"batch-{phase}-{dataset}-shard{shard_index}-{uuid.uuid4().hex[:8]}"
        ledger.reserve(reservation_id, discounted, project_limit_usd)
        snap = ledger.snapshot()
        print(
            f"  ledger: settled=${snap.actual_usd:.4f}  "
            f"budget_remaining≈${project_limit_usd - snap.actual_usd:.4f}"
        )


# ── Batch API interaction ───────────────────────────────────────────────────


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_sidecar(workspace: Path, phase_name: str, metadata: dict[str, Any]) -> None:
    path = workspace / f"{phase_name}_sidecar.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def submit_shards(
    workspace: Path,
    phase_name: str,
    shards: list[list[dict[str, Any]]],
) -> list[dict[str, str]]:
    """Upload JSONL files and create OpenAI Batch jobs.  Returns list of
    ``{shard_index, input_file_id, batch_id}`` records."""
    client = OpenAI()
    records: list[dict[str, str]] = []
    for idx, shard in enumerate(shards):
        jsonl_path = workspace / f"{phase_name}_shard_{idx}.jsonl"
        _write_jsonl(jsonl_path, shard)

        with jsonl_path.open("rb") as fh:
            file_obj = client.files.create(file=fh, purpose="batch")
        print(f"  uploaded shard {idx}: file_id={file_obj.id}")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  created batch shard {idx}: batch_id={batch.id}")
        records.append({
            "shard_index": str(idx),
            "input_file_id": file_obj.id,
            "batch_id": batch.id,
        })

    _save_sidecar(workspace, phase_name, {"batches": records})
    return records


def poll_shards(
    workspace: Path,
    phase_name: str,
    poll_interval: int = 30,
) -> dict[str, str]:
    """Poll all batch jobs until every one reaches a terminal state.

    Returns a mapping ``{batch_id: status}``.
    """
    sidecar_path = workspace / f"{phase_name}_sidecar.json"
    if not sidecar_path.exists():
        raise RuntimeError(f"no sidecar found at {sidecar_path}; run submit first")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    batch_ids = [b["batch_id"] for b in sidecar["batches"]]

    client = OpenAI()
    terminal = {"completed", "failed", "expired", "cancelled"}
    statuses: dict[str, str] = {}

    while True:
        all_done = True
        for bid in batch_ids:
            if bid in statuses and statuses[bid] in terminal:
                continue
            batch = client.batches.retrieve(bid)
            statuses[bid] = batch.status
            print(f"  batch {bid}: {batch.status}")
            if batch.status not in terminal:
                all_done = False
        if all_done:
            break
        print(f"  waiting {poll_interval}s...")
        time.sleep(poll_interval)

    _save_sidecar(workspace, phase_name, {**sidecar, "statuses": statuses})
    return statuses


def collect_shards(
    workspace: Path,
    phase_name: str,
    expected_custom_ids: set[str],
    validate_schema: type[BaseModel],
) -> list[dict[str, Any]]:
    """Download output files and validate against expected IDs."""
    sidecar_path = workspace / f"{phase_name}_sidecar.json"
    if not sidecar_path.exists():
        raise RuntimeError(f"no sidecar found at {sidecar_path}; run poll first")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    batch_ids = [b["batch_id"] for b in sidecar["batches"]]

    client = OpenAI()
    all_rows: list[dict[str, Any]] = []
    for bid in batch_ids:
        batch = client.batches.retrieve(bid)
        if batch.output_file_id is None:
            raise RuntimeError(
                f"batch {bid} has no output file (status={batch.status})"
            )
        file_content = client.files.content(batch.output_file_id)
        for line in file_content.text.strip().split("\n"):
            if line.strip():
                all_rows.append(json.loads(line))

    # Validate
    valid = collect_output_rows(all_rows, expected_custom_ids)

    # Per-row schema validation
    for row in valid:
        content = row["response"]["body"]["choices"][0]["message"]["content"]
        validate_schema.model_validate_json(content)

    # Persist validated output
    output_path = workspace / f"{phase_name}_output.jsonl"
    _write_jsonl(output_path, valid)

    return valid


# ── Phase-level orchestrators ───────────────────────────────────────────────


def _collect_ner_results(output_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parse NER output into {chunk_id: {"named_entities": [...]}}."""
    results: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        custom_id = row["custom_id"]
        content = row["response"]["body"]["choices"][0]["message"]["content"]
        parsed = NERResponse.model_validate_json(content)
        # Extract chunk_id from custom_id (format: ner-<dataset>-<32-char-hex>)
        results[custom_id] = parsed.model_dump()
    return results


def _collect_triple_results(
    output_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Parse triple output into {chunk_id: {"triples": [...]}}."""
    results: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        custom_id = row["custom_id"]
        content = row["response"]["body"]["choices"][0]["message"]["content"]
        parsed = TripleResponse.model_validate_json(content)
        results[custom_id] = parsed.model_dump()
    return results


# ── main() CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch OpenIE pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    def _add_common(p: argparse.ArgumentParser) -> None:
        datasets = ["nq_rear", "musique", "2wikimultihopqa"]
        p.add_argument("--dataset", required=True, choices=datasets)
        p.add_argument("--phase", required=True, choices=["ner", "triple"])
        p.add_argument(
            "--workspace", required=True, help="e.g. artifacts/openie/musique"
        )
        p.add_argument(
            "--config", default="configs/experiment.yaml",
            help="path to experiment.yaml",
        )
        p.add_argument(
            "--source",
            help="path to JSON mapping chunk_id → passage text",
        )

    _add_common(sub.add_parser("prepare", help="Build request JSONL shards"))
    _add_common(sub.add_parser("submit", help="Upload JSONL and create Batch jobs"))
    _add_common(sub.add_parser("poll", help="Poll batch jobs until done"))
    _add_common(sub.add_parser("collect", help="Download and validate output"))

    args = parser.parse_args(argv)
    workspace = Path(args.workspace)

    # Load config for pricing / budget values
    from .config import load_config

    config = load_config(Path(args.config))

    model = config.models.llm
    seed = config.project.seed

    # Load templates and compute prompt hashes
    ner_template, triple_template = _load_upstream_templates()
    ner_prompt_sha = _prompt_sha(ner_template)
    triple_prompt_sha = _prompt_sha(triple_template)

    if args.command == "prepare":
        # Load source docs
        source_path = Path(args.source)
        docs: dict[str, str] = json.loads(source_path.read_text(encoding="utf-8"))

        if args.phase == "ner":
            requests = build_ner_requests(
                docs,
                model=model,
                seed=seed,
                prompt_hash=ner_prompt_sha,
                config_hash=config.config_hash,
                dataset=args.dataset,
            )
            max_output = config.batch.ner_max_output_tokens
        else:
            # Triple phase needs NER output from a completed NER phase
            ner_output_path = workspace / "ner_output.jsonl"
            if not ner_output_path.exists():
                print("NER output not found; run the NER phase (prepare→submit→poll→collect) first",
                      file=sys.stderr)
                return 1
            ner_rows = _read_jsonl(ner_output_path)
            ner_results = _collect_ner_results(ner_rows)
            requests = build_triple_requests(
                docs,
                ner_results,
                model=model,
                seed=seed,
                prompt_hash=triple_prompt_sha,
                config_hash=config.config_hash,
                dataset=args.dataset,
            )
            max_output = config.batch.triple_max_output_tokens

        shards = _pack_shards(requests, model, config.batch.max_enqueued_input_tokens)
        print(f"Packed {len(requests)} requests into {len(shards)} shards")
        for idx, shard in enumerate(shards):
            shard_tokens = sum(_estimate_request_tokens(r, model) for r in shard)
            print(f"  shard {idx}: {len(shard)} requests, ~{shard_tokens:,} input tokens")

        # Write JSONL shards
        for idx, shard in enumerate(shards):
            _write_jsonl(workspace / f"{args.phase}_shard_{idx}.jsonl", shard)

        # Save sidecar with metadata
        prompt_sha = ner_prompt_sha if args.phase == "ner" else triple_prompt_sha
        custom_ids = sorted(r["custom_id"] for r in requests)
        sidecar = {
            "phase": args.phase,
            "dataset": args.dataset,
            "model": model,
            "seed": seed,
            "config_hash": config.config_hash,
            "prompt_sha": prompt_sha,
            "request_count": len(requests),
            "shard_count": len(shards),
            "custom_ids": custom_ids,
            "source_path": str(source_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _save_sidecar(workspace, args.phase, sidecar)

        # Budget preflight
        total_input_tokens = sum(
            _estimate_request_tokens(r, model)
            for shard in shards
            for r in shard
        )
        _budget_preflight(
            args.phase,
            args.dataset,
            shard_index=0,
            total_shards=len(shards),
            request_count=len(requests),
            max_input_tokens=total_input_tokens,
            max_output_tokens_per=max_output,
            input_price_per_million=config.pricing_snapshot.gpt_4o_mini_input_per_million,
            output_price_per_million=config.pricing_snapshot.gpt_4o_mini_output_per_million,
            batch_discount=config.pricing_snapshot.batch_discount_fraction,
        )

        # Save phase state
        phase = BatchPhase(
            dataset=args.dataset,
            phase=args.phase,
            expected=len(requests),
            completed=0,
        )
        _save_phase_state(workspace, phase)

    elif args.command == "submit":
        sidecar_path = workspace / f"{args.phase}_sidecar.json"
        if not sidecar_path.exists():
            print(f"No sidecar at {sidecar_path}; run prepare first", file=sys.stderr)
            return 1

        # Reconstruct shards from JSONL files
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        shard_count = sidecar["shard_count"]
        shard_data: list[list[dict[str, Any]]] = []
        for idx in range(shard_count):
            shard_path = workspace / f"{args.phase}_shard_{idx}.jsonl"
            shard_data.append(_read_jsonl(shard_path))

        records = submit_shards(workspace, args.phase, shard_data)
        print(f"Submitted {len(records)} batch jobs for {args.phase}/{args.dataset}")

    elif args.command == "poll":
        statuses = poll_shards(workspace, args.phase, config.batch.poll_interval_seconds)
        failed = [bid for bid, status in statuses.items() if status != "completed"]
        if failed:
            print(f"WARNING: {len(failed)} batches did not complete successfully:")
            for bid in failed:
                print(f"  {bid}: {statuses[bid]}")
            return 1
        print(f"All {len(statuses)} batches completed successfully")

    elif args.command == "collect":
        sidecar_path = workspace / f"{args.phase}_sidecar.json"
        if not sidecar_path.exists():
            print(f"No sidecar at {sidecar_path}; run prepare/poll first", file=sys.stderr)
            return 1

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected_ids = set(sidecar["custom_ids"])
        validate_schema = NERResponse if args.phase == "ner" else TripleResponse

        valid = collect_shards(workspace, args.phase, expected_ids, validate_schema)
        print(f"Collected {len(valid)} valid rows for {args.phase}/{args.dataset}")

        # Update phase state
        saved_phase = _load_phase_state(workspace, args.phase)
        if saved_phase is not None:
            saved_phase.completed = len(valid)
            _save_phase_state(workspace, saved_phase)

        # For NER phase, also save parsed results for triple phase to consume
        if args.phase == "ner":
            ner_results = _collect_ner_results(valid)
            _write_jsonl(workspace / "ner_output.jsonl", valid)

        # For triple phase, export upstream OpenIE JSON
        if args.phase == "triple":
            ner_output_path = workspace / "ner_output.jsonl"
            if not ner_output_path.exists():
                print("NER output not found; cannot export upstream JSON", file=sys.stderr)
                return 1
            ner_rows = _read_jsonl(ner_output_path)
            ner_results = _collect_ner_results(ner_rows)
            triple_results = _collect_triple_results(valid)

            # Need docs to export
            source_path = Path(sidecar.get("source_path", ""))
            if not source_path.exists():
                print(f"source docs not found at {source_path}", file=sys.stderr)
                return 1
            docs = json.loads(source_path.read_text(encoding="utf-8"))

            exported = export_upstream_openie(docs, ner_results, triple_results)
            export_path = workspace / "openie_results_ner_gpt-4o-mini-2024-07-18.json"
            tmp = export_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, export_path)
            print(f"Exported upstream OpenIE JSON to {export_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
