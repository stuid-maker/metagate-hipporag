"""Strict experiment configuration loading with canonical hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import DatasetId, MethodId


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(FrozenModel):
    name: Literal["metagate-hipporag2"]
    seed: Literal[20260711]
    upstream_commit: Literal["ad30fc3e2062202d9e975e32cd28212424a56ccb"]
    upstream_package_version: Literal["2.0.0-alpha.4"]
    data_revision: Literal["5ec05b38deecc3318bb432c69865959c56058990"]
    preprocessing_version: Literal[1]


class ModelConfig(FrozenModel):
    llm: Literal["gpt-4o-mini-2024-07-18"]
    embedding: Literal["text-embedding-3-large"]
    embedding_dimensions: Literal[3072]
    embedding_instruction_mode: Literal["upstream_ignored"]
    temperature: float = Field(ge=0.0, le=0.0)
    llm_seed: Literal[20260711]
    qa_max_output_tokens: int = Field(gt=0)
    gate_max_output_tokens: int = Field(gt=0)


class PromptConfig(FrozenModel):
    gate_file: Literal["configs/gate_prompt.json"]
    qa_file: Literal["configs/qa_prompt.json"]
    openie_source: Literal["pinned_upstream_templates"]


class RetrievalConfig(FrozenModel):
    output_top_k: Literal[5]
    linking_top_k: Literal[5]
    ppr_damping: float = Field(gt=0.0, lt=1.0)
    passage_node_weight: float = Field(ge=0.0, le=1.0)
    synonym_threshold: float = Field(ge=0.0, le=1.0)
    rrf_k: int = Field(gt=0)
    max_expansions: Literal[1]


class SamplingDatasetConfig(FrozenModel):
    id: DatasetId
    stratify_by: Literal["supporting_passage_count", "type"] | None


class SamplingConfig(FrozenModel):
    dev_per_dataset: int = Field(gt=0)
    test_per_dataset: int = Field(gt=0)
    datasets: tuple[SamplingDatasetConfig, ...]

    @model_validator(mode="after")
    def validate_dataset_plan(self) -> SamplingConfig:
        expected = (
            ("nq_rear", None),
            ("musique", "supporting_passage_count"),
            ("2wikimultihopqa", "type"),
        )
        actual = tuple((row.id, row.stratify_by) for row in self.datasets)
        if actual != expected:
            raise ValueError(f"sampling.datasets must equal {expected!r}")
        return self


class GateConfig(FrozenModel):
    threshold_candidates: tuple[float, ...]
    threshold_objective: Literal["balanced_accuracy"]
    threshold_target: Literal["recall_at_5_equals_1"]
    tuning_scope: Literal["pooled_equal_dataset_dev"]
    threshold_tie_breakers: tuple[
        Literal["lower_expansion_rate"], Literal["higher_threshold"]
    ]
    final_low_confidence_action: Literal["flag_abstain_and_keep_forced_answer"]

    @model_validator(mode="after")
    def validate_threshold_grid(self) -> GateConfig:
        expected = tuple(round(value / 100, 2) for value in range(50, 100, 5))
        if self.threshold_candidates != expected:
            raise ValueError(f"threshold_candidates must equal {expected!r}")
        return self


class StatisticsConfig(FrozenModel):
    bootstrap_resamples: int = Field(ge=1000)
    bootstrap_seed: Literal[20260711]
    confidence_level: float = Field(gt=0.0, lt=1.0)
    em_test: Literal["mcnemar_exact"]
    multiple_testing_correction: Literal["holm"]
    holm_family: Literal["three_datasets_x_three_primary_comparisons"]
    ece_bins: int = Field(ge=2)
    ece_binning: Literal["equal_frequency"]
    selective_risk_metric: Literal["one_minus_em"]
    noninferiority_margin_token_f1: float = Field(gt=0.0, le=0.05)


class EvaluationConfig(FrozenModel):
    retrieval_match: Literal["exact_upstream_title_newline_text"]
    answer_alias_aggregation: Literal["maximum"]
    llm_only_recall: Literal["not_applicable"]
    false_stop_denominator: Literal["first_retrieval_insufficient"]
    unnecessary_expansion_denominator: Literal["first_retrieval_sufficient"]
    calibration_scope: Literal["first_gate_all_examples"]


class PricingSnapshotConfig(FrozenModel):
    as_of: date
    currency: Literal["USD"]
    gpt_4o_mini_input_per_million: float = Field(ge=0.0)
    gpt_4o_mini_output_per_million: float = Field(ge=0.0)
    text_embedding_3_large_per_million: float = Field(ge=0.0)
    batch_discount_fraction: float = Field(ge=0.0, le=1.0)


class BudgetConfig(FrozenModel):
    smoke_max_actual_usd: float = Field(gt=0)
    project_max_actual_usd: float = Field(gt=0)
    actual_cache_hits_billable: Literal[False]
    method_equivalent_cache_hits_billable: Literal[True]
    batch_discount_scope: Literal["openie_batch_only"]


class BatchConfig(FrozenModel):
    max_enqueued_input_tokens: int = Field(gt=0, le=2_000_000)
    completion_window: Literal["24h"]
    endpoint: Literal["/v1/chat/completions"]
    max_requests_per_shard: int = Field(gt=0, le=50_000)
    poll_interval_seconds: int = Field(gt=0, le=30)
    ner_max_output_tokens: int = Field(gt=0, le=1024)
    triple_max_output_tokens: int = Field(gt=0, le=2048)


class ExperimentConfig(FrozenModel):
    project: ProjectConfig
    models: ModelConfig
    prompts: PromptConfig
    retrieval: RetrievalConfig
    sampling: SamplingConfig
    methods: tuple[MethodId, ...]
    gate: GateConfig
    statistics: StatisticsConfig
    evaluation: EvaluationConfig
    pricing_snapshot: PricingSnapshotConfig
    budget: BudgetConfig
    batch: BatchConfig
    config_hash: str = ""

    @model_validator(mode="after")
    def validate_methods_and_budget(self) -> ExperimentConfig:
        expected: tuple[MethodId, ...] = (
            "llm_only",
            "dense_rag",
            "hipporag2",
            "always_expand",
            "metagate",
        )
        if self.methods != expected:
            raise ValueError(f"methods must equal {expected!r}")
        if self.budget.smoke_max_actual_usd > self.budget.project_max_actual_usd:
            raise ValueError("smoke budget exceeds project budget")
        return self


def canonical_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat(),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    digest = canonical_hash(raw)
    return ExperimentConfig.model_validate({**raw, "config_hash": digest})


def effective_config_hash(
    config: ExperimentConfig, frozen_input_hashes: dict[str, str]
) -> str:
    required = {
        "compatibility_patch",
        "data_manifest",
        "gate_prompt",
        "qa_prompt",
        "openie_ner_prompt",
        "openie_triple_prompt",
    }
    if set(frozen_input_hashes) != required:
        raise ValueError(
            f"frozen input hash keys must equal {sorted(required)!r}"
        )
    if any(len(value) != 64 for value in frozen_input_hashes.values()):
        raise ValueError("every frozen input hash must be full SHA-256")
    return canonical_hash(
        {
            "base_config_hash": config.config_hash,
            "frozen_input_hashes": frozen_input_hashes,
        }
    )
