"""Shared Pydantic contracts for examples, traces, gates, usage, and results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DatasetId = Literal["nq_rear", "musique", "2wikimultihopqa"]
MethodId = Literal["llm_only", "dense_rag", "hipporag2", "always_expand", "metagate"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Example(StrictModel):
    dataset: DatasetId
    example_id: str
    question: str
    gold_answers: list[str]
    gold_docs: list[str]
    stratum: str


class RetrievedPassage(StrictModel):
    chunk_id: str
    text: str
    score: float
    rank: int = Field(ge=1)


class Usage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    embedding_tokens: int = Field(default=0, ge=0)
    observed_latency_seconds: float = Field(default=0.0, ge=0.0)
    method_equivalent_latency_seconds: float = Field(default=0.0, ge=0.0)
    cache_hit: bool = False
    actual_usd: float = Field(default=0.0, ge=0.0)
    method_equivalent_usd: float = Field(default=0.0, ge=0.0)


class LedgerEntry(StrictModel):
    event_id: str
    reservation_id: str
    stage: Literal[
        "openie_ner", "openie_triple", "embedding", "retrieval_llm", "gate", "qa"
    ]
    dataset: DatasetId | None = None
    example_id: str | None = None
    method: MethodId | None = None
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    embedding_tokens: int = Field(default=0, ge=0)
    cache_hit: bool
    batch_discount_applied: bool
    actual_usd: float = Field(default=0.0, ge=0.0)
    method_equivalent_usd: float = Field(default=0.0, ge=0.0)
    observed_latency_seconds: float = Field(default=0.0, ge=0.0)
    method_equivalent_latency_seconds: float = Field(default=0.0, ge=0.0)


class LedgerSnapshot(StrictModel):
    sequence: int = Field(ge=0)
    actual_usd: float = Field(ge=0.0)
    method_equivalent_usd: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    embedding_tokens: int = Field(ge=0)
    observed_latency_seconds: float = Field(ge=0.0)
    method_equivalent_latency_seconds: float = Field(ge=0.0)


class RetrievalTrace(StrictModel):
    retrieval_query: str
    passages: list[RetrievedPassage]
    facts_before_filter: list[tuple[str, str, str]]
    facts_after_filter: list[tuple[str, str, str]]
    used_dense_fallback: bool
    filter_error: str | None = None
    usage: Usage = Usage()


class GateDecision(StrictModel):
    evidence_sufficient_probability: float = Field(ge=0.0, le=1.0)
    missing_information: str
    retrieval_rewrite: str
    rationale_summary: str

    @field_validator("retrieval_rewrite")
    @classmethod
    def rewrite_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retrieval_rewrite must not be empty")
        return value.strip()


class MethodResult(StrictModel):
    run_id: str
    method: MethodId
    example: Example
    first_retrieval: RetrievalTrace | None = None
    second_retrieval: RetrievalTrace | None = None
    fused_passages: list[RetrievedPassage]
    answer: str
    gate_decisions: list[GateDecision]
    expanded: bool
    abstain_flag: bool
    usage: Usage
    errors: list[str]
