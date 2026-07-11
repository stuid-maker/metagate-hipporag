# MetaGate-HippoRAG 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, evaluate, and report a reproducible MetaGate extension to pinned HippoRAG 2 on NQ, MuSiQue, and 2Wiki, then deliver the course paper and narrated PPT by 2026-07-31.

**Architecture:** Keep the official HippoRAG checkout pinned and externally patched only for Windows/API-only imports. A project-owned bridge exposes traceable dense and graph retrieval; a cached OpenAI layer, two-stage Batch OpenIE state machine, MetaGate controller, resumable runner, and frozen statistics pipeline sit around it. All final numbers are regenerated from immutable run manifests, never copied by hand.

**Tech Stack:** Python 3.10, uv, Pydantic 2, OpenAI Python SDK 1.91.1, GPT-4o mini snapshot, text-embedding-3-large, PyTorch 2.5.1 CUDA 12.4, python-igraph, pandas/Parquet, pytest, statsmodels, matplotlib/seaborn, Word/PPT artifact skills, Windows PowerShell/System.Speech.

---

## Locked file map

| Path | Responsibility |
|---|---|
| `scripts/bootstrap_upstream.py` | Clone, verify, patch, and install the exact HippoRAG SHA |
| `patches/hipporag-openai-only.patch` | Lazy-import optional local model providers without changing retrieval semantics |
| `src/metagate_hipporag/models.py` | Shared Pydantic contracts for examples, traces, gates, usage, and results |
| `src/metagate_hipporag/config.py` | Strict YAML loading, environment checks, and stable configuration hashes |
| `src/metagate_hipporag/provenance.py` | SHA-256, atomic files, manifests, JSONL resume indexes, and cost ledger |
| `src/metagate_hipporag/data.py` | Pinned downloads, schema adapters, gold normalization, and fixed stratified splits |
| `src/metagate_hipporag/openai_client.py` | Structured completion cache, retries, usage, latency, and secret redaction |
| `src/metagate_hipporag/batch_openie.py` | NER → triple Batch phases, shard state, validation, recovery, and upstream JSON export |
| `src/metagate_hipporag/embedding.py` | Persistent OpenAI embedding cache, usage ledger, and HippoRAG injection |
| `src/metagate_hipporag/hipporag_adapter.py` | Fingerprinted indexes, dense/graph retrieval bridge, trace capture, and QA |
| `src/metagate_hipporag/fusion.py` | Deterministic chunk-ID Reciprocal Rank Fusion |
| `src/metagate_hipporag/metagate.py` | Fixed zero-shot prompt, gate parsing, threshold tuning, and one-expansion policy |
| `src/metagate_hipporag/methods.py` | Five method conditions, shared-call accounting, and resumable per-query execution |
| `src/metagate_hipporag/evaluation.py` | Retrieval, QA, calibration, coverage, risk, and efficiency metrics |
| `src/metagate_hipporag/statistics.py` | Paired bootstrap, exact McNemar, Holm correction, tables, and figures |
| `src/metagate_hipporag/cli.py` | Stable commands for every pipeline stage |
| `tests/` | Unit, upstream-contract, resume, and opt-in live API tests |
| `results/` | Frozen, manifest-linked CSV tables and publication figures |
| `paper/` / `slides/` | Final research report, bibliography, deck, narration script, and embedded audio |

## Task 0: Recover repository writes and lock the environment

**Files:**
- Modify: `uv.lock` (generated)
- Verify: `.gitignore`, `pyproject.toml`, `configs/experiment.yaml`

- [ ] **Step 1: Run the existing scaffold validation**

Run from a terminal that has normal write permission to `.git` and internet access:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:PYTHONHASHSEED = '20260711'
D:\anaconda\python.exe -c "import pathlib,tomllib,yaml; tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); c=yaml.safe_load(pathlib.Path('configs/experiment.yaml').read_text(encoding='utf-8')); assert len(c['methods']) == 5"
```

Expected: command exits 0 with no output.

- [ ] **Step 2: Generate and verify the dependency lock**

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:PYTHONHASHSEED = '20260711'
uv lock --python 3.10
uv sync --python 3.10 --group dev --group analysis
uv run python -c "import sys, torch; assert sys.version_info[:2] == (3, 10); print(torch.__version__, torch.cuda.is_available())"
```

Expected: `uv.lock` is created; Python prints `2.5.1` and `True`. If CUDA prints `False`, stop before indexing and verify that uv selected the explicit `cu124` source in `pyproject.toml`.

- [ ] **Step 3: Commit the approved scaffold and design**

```powershell
git add .env.example .gitattributes .gitignore README.md pyproject.toml uv.lock configs data artifacts docs paper patches results scripts slides src tests third_party
git diff --cached --check
git commit -m "docs: establish MetaGate research design"
```

Expected: one commit is created; the course PDF remains unchanged and ignored.

## Task 1: Bootstrap the exact upstream HippoRAG source

**Files:**
- Create: `scripts/bootstrap_upstream.py`
- Create: `patches/hipporag-openai-only.patch`
- Create: `patches/hipporag-openai-only.patch.sha256`
- Create: `src/metagate_hipporag/__init__.py`
- Create: `tests/test_bootstrap_upstream.py`

- [ ] **Step 1: Write the failing bootstrap tests**

```python
# tests/test_bootstrap_upstream.py
from pathlib import Path

from scripts.bootstrap_upstream import PATCH_FILE, UPSTREAM_COMMIT, verify_checkout


def test_upstream_commit_is_exact_sha() -> None:
    assert UPSTREAM_COMMIT == "ad30fc3e2062202d9e975e32cd28212424a56ccb"
    assert len(UPSTREAM_COMMIT) == 40


def test_upstream_package_version_is_frozen() -> None:
    assert UPSTREAM_PACKAGE_VERSION == "2.0.0-alpha.4"


def test_patch_is_tracked_and_semantics_limited() -> None:
    patch = PATCH_FILE.read_text(encoding="utf-8")
    assert "embedding_model/__init__.py" in patch
    assert "llm/__init__.py" in patch
    assert "run_ppr" not in patch
    assert "graph_search_with_fact_entities" not in patch


def test_verify_checkout_rejects_wrong_sha(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    try:
        verify_checkout(tmp_path, actual_commit="0" * 40)
    except RuntimeError as exc:
        assert "unexpected HippoRAG commit" in str(exc)
    else:
        raise AssertionError("wrong SHA was accepted")
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/test_bootstrap_upstream.py -q
```

Expected: collection fails because `scripts.bootstrap_upstream` does not exist.

- [ ] **Step 3: Add the compatibility patch**

```diff
diff --git a/src/hipporag/embedding_model/__init__.py b/src/hipporag/embedding_model/__init__.py
index 8d27583..5397eae 100644
--- a/src/hipporag/embedding_model/__init__.py
+++ b/src/hipporag/embedding_model/__init__.py
@@ -1,11 +1,4 @@
-from .Contriever import ContrieverModel
 from .base import EmbeddingConfig, BaseEmbeddingModel
-from .GritLM import GritLMEmbeddingModel
-from .NVEmbedV2 import NVEmbedV2EmbeddingModel
-from .OpenAI import OpenAIEmbeddingModel
-from .Cohere import CohereEmbeddingModel
-from .Transformers import TransformersEmbeddingModel
-from .VLLM import VLLMEmbeddingModel

 from ..utils.logging_utils import get_logger

@@ -14,17 +7,31 @@ logger = get_logger(__name__)

 def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
     if "GritLM" in embedding_model_name:
+        from .GritLM import GritLMEmbeddingModel
+
         return GritLMEmbeddingModel
     elif "NV-Embed-v2" in embedding_model_name:
+        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
+
         return NVEmbedV2EmbeddingModel
     elif "contriever" in embedding_model_name:
+        from .Contriever import ContrieverModel
+
         return ContrieverModel
     elif "text-embedding" in embedding_model_name:
+        from .OpenAI import OpenAIEmbeddingModel
+
         return OpenAIEmbeddingModel
     elif "cohere" in embedding_model_name:
+        from .Cohere import CohereEmbeddingModel
+
         return CohereEmbeddingModel
     elif embedding_model_name.startswith("Transformers/"):
+        from .Transformers import TransformersEmbeddingModel
+
         return TransformersEmbeddingModel
     elif embedding_model_name.startswith("VLLM/"):
+        from .VLLM import VLLMEmbeddingModel
+
         return VLLMEmbeddingModel
-    assert False, f"Unknown embedding model name: {embedding_model_name}"
\ No newline at end of file
+    assert False, f"Unknown embedding model name: {embedding_model_name}"
diff --git a/src/hipporag/llm/__init__.py b/src/hipporag/llm/__init__.py
index 93b2f04..6097884 100644
--- a/src/hipporag/llm/__init__.py
+++ b/src/hipporag/llm/__init__.py
@@ -5,8 +5,6 @@ from ..utils.config_utils import BaseConfig

 from .openai_gpt import CacheOpenAI
 from .base import BaseLLM
-from .bedrock_llm import BedrockLLM
-from .transformers_llm import TransformersLLM


 logger = get_logger(__name__)
@@ -17,10 +15,14 @@ def _get_llm_class(config: BaseConfig):
         os.environ['OPENAI_API_KEY'] = 'sk-'

     if config.llm_name.startswith('bedrock'):
+        from .bedrock_llm import BedrockLLM
+
         return BedrockLLM(config)

     if config.llm_name.startswith('Transformers/'):
+        from .transformers_llm import TransformersLLM
+
         return TransformersLLM(config)
     
     return CacheOpenAI.from_experiment_config(config)
-    
\ No newline at end of file
+    
```

Save this exact unified diff as `patches/hipporag-openai-only.patch`. Generate and commit `patches/hipporag-openai-only.patch.sha256`; the bootstrap test must run `git apply --check`, assert the touched-file set is exactly the two paths above, and reject any patch hunk mentioning retrieval, PPR, OpenIE, or QA files.

- [ ] **Step 4: Implement the bootstrap script**

```python
# scripts/bootstrap_upstream.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "third_party" / "HippoRAG"
PATCH_FILE = REPO_ROOT / "patches" / "hipporag-openai-only.patch"
UPSTREAM_URL = "https://github.com/OSU-NLP-Group/HippoRAG.git"
UPSTREAM_COMMIT = "ad30fc3e2062202d9e975e32cd28212424a56ccb"
UPSTREAM_PACKAGE_VERSION = "2.0.0-alpha.4"
PATCH_MARKER = UPSTREAM_DIR / ".metagate-openai-only-patch"


def run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(args), cwd=cwd, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def verify_checkout(path: Path, actual_commit: str | None = None) -> None:
    if not (path / ".git").exists():
        raise RuntimeError(f"missing git checkout: {path}")
    actual = actual_commit or run("git", "rev-parse", "HEAD", cwd=path)
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"unexpected HippoRAG commit: {actual}")


def bootstrap(check_only: bool) -> None:
    if not UPSTREAM_DIR.exists():
        if check_only:
            raise RuntimeError(f"missing upstream checkout: {UPSTREAM_DIR}")
        UPSTREAM_DIR.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", UPSTREAM_URL, str(UPSTREAM_DIR))
        run("git", "checkout", "--detach", UPSTREAM_COMMIT, cwd=UPSTREAM_DIR)
    verify_checkout(UPSTREAM_DIR)
    if not PATCH_MARKER.exists():
        if check_only:
            raise RuntimeError("compatibility patch has not been applied")
        run("git", "apply", "--check", str(PATCH_FILE), cwd=UPSTREAM_DIR)
        run("git", "apply", str(PATCH_FILE), cwd=UPSTREAM_DIR)
        PATCH_MARKER.write_text(UPSTREAM_COMMIT + "\n", encoding="utf-8")
    if not check_only:
        run(
            "uv", "pip", "install", "--no-deps", "--editable", str(UPSTREAM_DIR),
            cwd=REPO_ROOT,
        )
    verify_checkout(UPSTREAM_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bootstrap(check_only=args.check)
    print(f"HippoRAG ready at {UPSTREAM_COMMIT}")


if __name__ == "__main__":
    sys.exit(main())
```

Before considering the script complete, add `verify_patch_state()`: compare the patch bytes with `patches/hipporag-openai-only.patch.sha256`; parse upstream `setup.py` and assert `2.0.0-alpha.4`; require `git diff --name-only` to equal exactly the two lazy-import files; run `git diff --check`; and compare the applied diff content with the tracked patch. Call it on normal and `--check` paths. Store upstream SHA, patch SHA, package version, and UTC application time in the marker as JSON; marker presence alone is never accepted as proof.

```python
# src/metagate_hipporag/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 5: Run tests and bootstrap**

```powershell
uv run pytest tests/test_bootstrap_upstream.py -q
uv run python scripts/bootstrap_upstream.py
uv run python scripts/bootstrap_upstream.py --check
uv run python -c "from hipporag import HippoRAG; print('HippoRAG import OK')"
```

Expected: all bootstrap tests pass; both bootstrap commands print the pinned SHA, version, and patch SHA; import prints `HippoRAG import OK`.

- [ ] **Step 6: Commit**

```powershell
git add patches/hipporag-openai-only.patch patches/hipporag-openai-only.patch.sha256 scripts/bootstrap_upstream.py src/metagate_hipporag/__init__.py tests/test_bootstrap_upstream.py
git commit -m "build: pin API-only HippoRAG upstream"
```

## Task 2: Define strict contracts, configuration, and provenance

**Files:**
- Create: `src/metagate_hipporag/models.py`
- Create: `src/metagate_hipporag/config.py`
- Create: `src/metagate_hipporag/provenance.py`
- Create: `tests/test_config_and_provenance.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_config_and_provenance.py
from pathlib import Path

from metagate_hipporag.config import load_config
from metagate_hipporag.models import GateDecision, RetrievedPassage
from metagate_hipporag.provenance import atomic_write_json, sha256_file


def test_config_is_frozen_and_hash_is_stable() -> None:
    first = load_config(Path("configs/experiment.yaml"))
    second = load_config(Path("configs/experiment.yaml"))
    assert first.config_hash == second.config_hash
    assert first.models.llm == "gpt-4o-mini-2024-07-18"
    assert first.sampling.dev_per_dataset == 100
    assert first.sampling.test_per_dataset == 300
    assert first.budget.project_max_actual_usd == 18.0


def test_gate_probability_is_bounded() -> None:
    GateDecision(
        evidence_sufficient_probability=0.5,
        missing_information="a bridge fact",
        retrieval_rewrite="bridge entity relation",
        rationale_summary="One supporting link is absent.",
    )
    try:
        GateDecision(
            evidence_sufficient_probability=1.1,
            missing_information="x",
            retrieval_rewrite="y",
            rationale_summary="z",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range probability accepted")


def test_passage_requires_stable_chunk_id() -> None:
    passage = RetrievedPassage(chunk_id="chunk-abc", text="Title\nBody", score=0.2, rank=1)
    assert passage.chunk_id == "chunk-abc"


def test_atomic_json_and_sha(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert len(sha256_file(target)) == 64
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/test_config_and_provenance.py -q
```

Expected: collection fails because the three project modules do not exist.

- [ ] **Step 3: Implement the shared Pydantic models**

Create `src/metagate_hipporag/models.py` with the exact models from the design and these constraints:

```python
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
    stage: Literal["openie_ner", "openie_triple", "embedding", "retrieval_llm", "gate", "qa"]
    dataset: DatasetId | None = None
    example_id: str | None = None
    method: MethodId | None = None
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    embedding_tokens: int = Field(default=0, ge=0)
    cache_hit: bool
    batch_discount_applied: bool
    actual_usd: float = Field(ge=0.0)
    method_equivalent_usd: float = Field(ge=0.0)
    observed_latency_seconds: float = Field(ge=0.0)
    method_equivalent_latency_seconds: float = Field(ge=0.0)


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
```

- [ ] **Step 4: Implement strict config loading**

In `src/metagate_hipporag/config.py`, define nested frozen Pydantic models matching every key in `configs/experiment.yaml`, including the Batch limits already frozen in the scaffold, then implement canonical hashing:

```python
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
    def validate_dataset_plan(self) -> "SamplingConfig":
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
    def validate_threshold_grid(self) -> "GateConfig":
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
    def validate_methods_and_budget(self) -> "ExperimentConfig":
        expected = ("llm_only", "dense_rag", "hipporag2", "always_expand", "metagate")
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
        raise ValueError(f"frozen input hash keys must equal {sorted(required)!r}")
    if any(len(value) != 64 for value in frozen_input_hashes.values()):
        raise ValueError("every frozen input hash must be full SHA-256")
    return canonical_hash(
        {
            "base_config_hash": config.config_hash,
            "frozen_input_hashes": frozen_input_hashes,
        }
    )
```

The finished loader rejects unknown keys, altered model IDs, reordered/duplicated methods or datasets, an altered threshold grid, an invalid one-expansion policy, and a smoke budget larger than the total budget. The base hash covers YAML bytes semantically; the effective hash additionally covers every prompt, the compatibility patch, and the data manifest, so changing any of them forces a new run ID.

- [ ] **Step 5: Implement provenance primitives**

```python
# src/metagate_hipporag/provenance.py
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock")
    with lock, path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
```

Import `FileLock` from `filelock`. Add `read_jsonl_recover_tail(path)`: parse every newline-terminated record; if and only if the final non-newline-terminated fragment is invalid JSON, copy the original to `<name>.corrupt.<UTC timestamp>`, truncate that tail under the same lock, and return the valid rows. Invalid JSON in any earlier record or a duplicate immutable event is a hard failure. Completion-key interpretation is intentionally deferred to the typed runner in Task 9.

- [ ] **Step 6: Implement the unified actual/method-equivalent cost ledger**

Add strict `LedgerEntry` and `LedgerSnapshot` models, then a SQLite-backed `UsageLedger` in `provenance.py` with this public contract:

```python
class UsageLedger:
    def reserve(self, reservation_id: str, upper_bound_actual_usd: float, limit_usd: float) -> None: ...
    def append(self, entry: LedgerEntry) -> None: ...
    def settle(self, reservation_id: str, entries: list[LedgerEntry]) -> None: ...
    def release(self, reservation_id: str, reason: str) -> None: ...
    def snapshot(self) -> LedgerSnapshot: ...
    def delta(self, before: LedgerSnapshot, after: LedgerSnapshot) -> Usage: ...
```

Use SQLite tables `ledger_entries(event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)` and `reservations(reservation_id TEXT PRIMARY KEY, upper_bound_actual_usd REAL NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL)`. `reserve()` runs under `BEGIN IMMEDIATE` and rejects when settled actual cost + all active reservations + the new upper bound exceeds the configured project limit. `append()` is idempotent only when the existing canonical payload is byte-identical; a conflicting duplicate is an error. `settle()` atomically appends validated entries and closes the reservation. Actual cache hits have zero `actual_usd`; every method that consumes a shared/cached node retains its independently priced `method_equivalent_usd`. Store separate `observed_latency_seconds` and `method_equivalent_latency_seconds`; never reuse historical network latency as current cache latency.

Add tests for concurrent reservation exclusion, pending-Batch budget protection, idempotent identical append, conflicting duplicate rejection, actual cache-hit cost zero, nonzero method-equivalent cached cost, and snapshot/delta arithmetic. Every paid Chat, Embedding, and Batch entry point in later tasks must reserve before calling the API and settle or explicitly release afterward.

- [ ] **Step 7: Run tests, type checks, and commit**

```powershell
uv run pytest tests/test_config_and_provenance.py -q
uv run ruff check src/metagate_hipporag/models.py src/metagate_hipporag/config.py src/metagate_hipporag/provenance.py tests/test_config_and_provenance.py
uv run mypy src/metagate_hipporag/models.py src/metagate_hipporag/config.py src/metagate_hipporag/provenance.py
git add src/metagate_hipporag tests/test_config_and_provenance.py
git commit -m "feat: add strict experiment contracts and provenance"
```

Expected: all tests, Ruff, and mypy pass.

## Task 3: Download, normalize, and freeze the three data splits

**Files:**
- Create: `src/metagate_hipporag/data.py`
- Create: `tests/fixtures/musique_small.json`
- Create: `tests/fixtures/twowiki_small.json`
- Create: `tests/fixtures/nq_small.json`
- Create: `tests/test_data.py`
- Generate: `data/manifest.json`, `data/splits/*.json`

- [ ] **Step 1: Write failing normalization and split tests**

```python
# tests/test_data.py
from collections import Counter
from pathlib import Path

from metagate_hipporag.data import deterministic_split, normalize_example


def test_musique_normalization() -> None:
    raw = {
        "id": "2hop__1_2",
        "question": "Q?",
        "answer": "A",
        "answer_aliases": ["Alias"],
        "paragraphs": [
            {"title": "Gold", "paragraph_text": "Evidence", "is_supporting": True},
            {"title": "Noise", "paragraph_text": "Other", "is_supporting": False},
        ],
    }
    example = normalize_example("musique", raw)
    assert example.example_id == "2hop__1_2"
    assert example.gold_answers == ["A", "Alias"]
    assert example.gold_docs == ["Gold\nEvidence"]
    assert example.stratum == "1"


def test_split_is_disjoint_reproducible_and_stratified() -> None:
    rows = [
        normalize_example(
            "musique",
            {
                "id": f"{2 + i % 3}hop__{i}",
                "question": f"q{i}",
                "answer": f"a{i}",
                "paragraphs": [
                    {"title": f"g{i}-{j}", "paragraph_text": "e", "is_supporting": True}
                    for j in range(2 + i % 3)
                ],
            },
        )
        for i in range(60)
    ]
    first = deterministic_split(
        rows, dev_size=12, test_size=24, seed=20260711, source_sha256="0" * 64
    )
    second = deterministic_split(
        rows, dev_size=12, test_size=24, seed=20260711, source_sha256="0" * 64
    )
    assert first == second
    assert set(first["dev_ids"]).isdisjoint(first["test_ids"])
    strata = Counter(row.stratum for row in rows if row.example_id in first["test_ids"])
    assert set(strata) == {"2", "3", "4"}
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/test_data.py -q
```

Expected: import fails because `data.py` does not exist.

- [ ] **Step 3: Implement pinned downloads and normalization**

Use the Hugging Face revision `5ec05b3` and only these six files. `download_data()` must stream to a temporary path, verify nonzero length, hash it, atomically rename it, and write the manifest. Normalize IDs as follows: NQ uses `id`; MuSiQue uses `id`; 2Wiki uses `_id`. Normalize gold documents using the exact logic in pinned upstream `main.py::get_gold_docs`.

```python
# src/metagate_hipporag/data.py (core constants and dispatch)
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import DatasetId, Example
from .provenance import atomic_write_json, sha256_file

HF_REVISION = "5ec05b38deecc3318bb432c69865959c56058990"
HF_BASE = f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{HF_REVISION}"
DATASET_FILES = {
    dataset: (f"{dataset}.json", f"{dataset}_corpus.json")
    for dataset in ("nq_rear", "musique", "2wikimultihopqa")
}


def document_text(title: str, text: str) -> str:
    return f"{title}\n{text}"


def upstream_gold_docs(dataset: DatasetId, raw: dict[str, Any]) -> list[str]:
    if "supporting_facts" in raw:
        supporting_titles = {title for title, _ in raw["supporting_facts"]}
        gold = [
            document_text(title, " ".join(sentences))
            for title, sentences in raw["context"]
            if title in supporting_titles
        ]
    elif "contexts" in raw:
        gold = [
            document_text(row["title"], row["text"])
            for row in raw["contexts"]
            if row["is_supporting"]
        ]
    elif "paragraphs" in raw:
        supporting = [row for row in raw["paragraphs"] if row.get("is_supporting", True)]
        gold = [
            document_text(
                row["title"],
                row["text"] if "text" in row else row["paragraph_text"],
            )
            for row in supporting
        ]
    else:
        raise ValueError(f"unsupported gold-document schema for {dataset}")
    return list(dict.fromkeys(gold))


def normalize_example(dataset: DatasetId, raw: dict[str, Any]) -> Example:
    gold = upstream_gold_docs(dataset, raw)
    answer_value = raw["answer"]
    answers = answer_value if isinstance(answer_value, list) else [answer_value]
    answers = [*answers, *raw.get("answer_aliases", [])]
    example_id = raw["id"] if "id" in raw else raw["_id"]
    if dataset == "musique":
        stratum = str(len(gold))
    elif dataset == "2wikimultihopqa":
        stratum = raw["type"]
    else:
        stratum = "simple"
    return Example(
        dataset=dataset,
        example_id=str(example_id),
        question=raw["question"],
        gold_answers=list(dict.fromkeys(answers)),
        gold_docs=gold,
        stratum=stratum,
    )
```

Implement `download_data()` around these constants using `httpx.Client(follow_redirects=True, timeout=60.0).stream()`, `raise_for_status()`, a same-directory `.part` file, `os.replace()`, and streaming SHA-256. The first successful download writes requested URL, final URL, ETag, full revision, byte count, SHA-256, and UTC time to `data/manifest.json`; every rerun verifies the local bytes and refuses a remote or local hash mismatch instead of silently refreshing. Add `main()` with the module command `prepare --config <path>`; it asserts 1,000 questions per source and corpus counts NQ 9,633 / MuSiQue 11,656 / 2Wiki 6,119, validates the three real schema fixtures, normalizes examples, freezes splits, and writes no corpus contents to Git.

- [ ] **Step 4: Implement proportional deterministic splits**

```python
def _allocate(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(group_sizes.values())
    if total < 0 or total > population or population == 0:
        raise ValueError("requested allocation exceeds population")
    raw = {name: total * size / population for name, size in group_sizes.items()}
    allocated = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - allocated[name]), name))
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def deterministic_split(
    rows: list[Example],
    dev_size: int,
    test_size: int,
    seed: int,
    source_sha256: str,
) -> dict[str, object]:
    if not rows or len(source_sha256) != 64:
        raise ValueError("rows and full source SHA-256 are required")
    datasets = {row.dataset for row in rows}
    if len(datasets) != 1:
        raise ValueError("a split may contain exactly one dataset")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("duplicate example IDs")
    groups: dict[str, list[Example]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item.example_id):
        groups[row.stratum].append(row)
    dataset = next(iter(datasets))
    expected_strata = {
        "nq_rear": {"simple"},
        "musique": {"2", "3", "4"},
        "2wikimultihopqa": {
            "compositional", "comparison", "bridge_comparison", "inference"
        },
    }[dataset]
    if set(groups) != expected_strata:
        raise ValueError(f"unexpected {dataset} strata: {sorted(groups)!r}")
    if dev_size + test_size > len(rows):
        raise ValueError("development and test sizes exceed population")
    dev_alloc = _allocate({key: len(value) for key, value in groups.items()}, dev_size)
    remaining_sizes = {key: len(value) - dev_alloc[key] for key, value in groups.items()}
    test_alloc = _allocate(remaining_sizes, test_size)
    rng = random.Random(seed)
    dev_ids: list[str] = []
    test_ids: list[str] = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        dev_count = dev_alloc[key]
        test_count = test_alloc[key]
        dev_ids.extend(row.example_id for row in group[:dev_count])
        test_ids.extend(row.example_id for row in group[dev_count : dev_count + test_count])
    if len(dev_ids) != dev_size or len(test_ids) != test_size:
        raise ValueError("unable to satisfy requested split sizes")
    if set(dev_ids) & set(test_ids):
        raise ValueError("development and test split overlap")
    return {
        "dataset": dataset,
        "data_revision": HF_REVISION,
        "source_sha256": source_sha256,
        "seed": seed,
        "dev_ids": sorted(dev_ids),
        "test_ids": sorted(test_ids),
        "stratum_counts": {
            key: {
                "population": len(groups[key]),
                "dev": dev_alloc[key],
                "test": test_alloc[key],
            }
            for key in sorted(groups)
        },
    }
```

- [ ] **Step 5: Run tests, download, freeze splits, and inspect counts**

```powershell
uv run pytest tests/test_data.py -q
uv run python -m metagate_hipporag.data prepare --config configs/experiment.yaml
uv run python -c "import json,pathlib; [print(p.name, len(json.loads(p.read_text(encoding='utf-8'))['dev_ids']), len(json.loads(p.read_text(encoding='utf-8'))['test_ids'])) for p in pathlib.Path('data/splits').glob('*.json')]"
```

Expected: tests pass; each split prints `100 300`; `data/manifest.json` records six URLs and six SHA-256 values.

- [ ] **Step 6: Commit only manifests and split IDs, not raw data**

```powershell
git add src/metagate_hipporag/data.py tests/fixtures tests/test_data.py data/manifest.json data/splits
git commit -m "feat: freeze three benchmark samples"
```

## Task 4: Add a cached, structured OpenAI client and usage ledger

**Files:**
- Create: `src/metagate_hipporag/openai_client.py`
- Create: `tests/test_openai_client.py`

- [ ] **Step 1: Write a failing cache and schema test**

```python
# tests/test_openai_client.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/test_openai_client.py -q
```

Expected: import fails because `openai_client.py` does not exist.

- [ ] **Step 3: Implement the cache key, SQLite table, and injectable backend**

```python
# src/metagate_hipporag/openai_client.py
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import Usage
from .provenance import UsageLedger

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class RawCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class StructuredCompletion(Generic[T]):
    value: T
    usage: Usage


def _cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CachedStructuredClient:
    def __init__(
        self,
        cache_path: Path,
        invoke: Callable[..., RawCompletion] | None = None,
        input_price_per_million: float = 0.15,
        output_price_per_million: float = 0.60,
        ledger: UsageLedger | None = None,
        project_limit_usd: float = 18.0,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self.invoke = invoke or self._invoke_openai
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.ledger = ledger
        self.project_limit_usd = project_limit_usd
        if invoke is None and ledger is None:
            raise ValueError("a production OpenAI client requires UsageLedger")
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS completions ("
                "key TEXT PRIMARY KEY, custom_id TEXT NOT NULL, response TEXT NOT NULL, "
                "usage TEXT NOT NULL)"
            )

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        reraise=True,
    )
    def _invoke_openai(**kwargs: Any) -> RawCompletion:
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

    def complete(
        self,
        *,
        custom_id: str,
        model: str,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_completion_tokens: int,
        seed: int,
        temperature: float,
    ) -> StructuredCompletion[T]:
        started = time.perf_counter()
        payload = {
            "base_url": "https://api.openai.com/v1",
            "model": model,
            "messages": messages,
            "schema": response_model.model_json_schema(),
            "max_completion_tokens": max_completion_tokens,
            "seed": seed,
            "temperature": temperature,
        }
        key = _cache_key(payload)
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
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
            return StructuredCompletion(
                value=response_model.model_validate_json(row[0]),
                usage=cached_usage,
            )
        raw = self.invoke(
            custom_id=custom_id,
            model=model,
            messages=messages,
            response_model=response_model,
            max_completion_tokens=max_completion_tokens,
            seed=seed,
            temperature=temperature,
        )
        value = response_model.model_validate_json(raw.content)
        actual = (
            raw.prompt_tokens * self.input_price + raw.completion_tokens * self.output_price
        ) / 1_000_000
        usage = Usage(
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            observed_latency_seconds=raw.latency_seconds,
            method_equivalent_latency_seconds=raw.latency_seconds,
            cache_hit=False,
            actual_usd=actual,
            method_equivalent_usd=actual,
        )
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                "INSERT INTO completions(key, custom_id, response, usage) VALUES (?, ?, ?, ?)",
                (key, custom_id, value.model_dump_json(), usage.model_dump_json()),
            )
            connection.commit()
        return StructuredCompletion(value=value, usage=usage)
```

The cached result preserves `method_equivalent_usd` for fair method accounting but sets actual incremental cost to zero. `custom_id` is audit metadata and is deliberately excluded from the cache key, so identical requests shared across methods reuse one billable call. Before every cache miss, estimate input tokens from the canonical messages/schema plus the configured output ceiling, reserve that maximum in `UsageLedger`, invoke once, and atomically settle a `LedgerEntry`; release only after a confirmed pre-submission/local failure. Each cache hit appends an idempotent zero-actual-cost consumption event keyed by `(custom_id, cache_key)` with nonzero method-equivalent cost. Tests may omit the ledger only when an injected fake `invoke` is supplied.

- [ ] **Step 4: Add malformed-output and secret-redaction tests**

Add tests that assert invalid JSON raises `ValidationError` immediately, the cache key changes when model/schema/prompt changes, and neither `OPENAI_API_KEY` nor an `Authorization` header is ever serialized. Retry only API connection, timeout, and rate-limit exceptions up to three attempts; authentication, permission, bad-request, and schema-validation failures must fail immediately.

- [ ] **Step 5: Run checks and commit**

```powershell
uv run pytest tests/test_openai_client.py -q
uv run ruff check src/metagate_hipporag/openai_client.py tests/test_openai_client.py
uv run mypy src/metagate_hipporag/openai_client.py
git add src/metagate_hipporag/openai_client.py tests/test_openai_client.py
git commit -m "feat: add cached structured OpenAI client"
```

## Task 5: Implement resumable two-stage Batch OpenIE

**Files:**
- Create: `src/metagate_hipporag/batch_openie.py`
- Create: `tests/fixtures/batch_ner_output.jsonl`
- Create: `tests/fixtures/batch_triple_output.jsonl`
- Create: `tests/test_batch_openie.py`
- Verify: `configs/experiment.yaml`
- Create: `tests/fixtures/openie_smoke_docs.json`

- [ ] **Step 1: Verify the frozen deterministic Batch limits**

Confirm that the scaffold already contains exactly:

```yaml
batch:
  max_enqueued_input_tokens: 1500000
  completion_window: 24h
  endpoint: /v1/chat/completions
  max_requests_per_shard: 5000
  poll_interval_seconds: 30
  ner_max_output_tokens: 512
  triple_max_output_tokens: 1024
```

`BatchConfig` was defined in Task 2. Add a test that unknown keys, intervals above 30 seconds, and enqueued-token limits above 2,000,000 are rejected. The configured shard ceiling is deliberately conservative; if a submit response reports a lower account limit, stop, release only the unsubmitted reservation, reduce the configuration in a new committed run, and never silently resize a frozen shard set.

- [ ] **Step 2: Write failing state-machine tests**

```python
# tests/test_batch_openie.py
from pathlib import Path

from metagate_hipporag.batch_openie import (
    BatchPhase,
    build_ner_requests,
    collect_output_rows,
    export_upstream_openie,
)


def test_ner_requests_have_stable_unique_custom_ids() -> None:
    docs = {"chunk-a": "Title A\nBody A", "chunk-b": "Title B\nBody B"}
    rows = build_ner_requests(docs, model="gpt-4o-mini-2024-07-18", seed=20260711)
    assert len(rows) == 2
    assert len({row["custom_id"] for row in rows}) == 2
    assert all(row["url"] == "/v1/chat/completions" for row in rows)
    assert all(row["body"]["response_format"]["type"] == "json_schema" for row in rows)


def test_triple_phase_cannot_start_before_complete_ner(tmp_path: Path) -> None:
    phase = BatchPhase(dataset="musique", phase="ner", expected=2, completed=1)
    try:
        phase.require_complete()
    except RuntimeError as exc:
        assert "1/2" in str(exc)
    else:
        raise AssertionError("incomplete NER phase was accepted")


def test_duplicate_or_failed_output_is_rejected() -> None:
    rows = [
        {"custom_id": "a", "response": {"status_code": 200, "body": {}}, "error": None},
        {"custom_id": "a", "response": {"status_code": 200, "body": {}}, "error": None},
    ]
    try:
        collect_output_rows(rows, expected_ids={"a"})
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate output was accepted")
```

- [ ] **Step 3: Run the tests to verify they fail**

```powershell
uv run pytest tests/test_batch_openie.py -q
```

Expected: import fails because `batch_openie.py` does not exist.

- [ ] **Step 4: Implement schemas, request construction, and shard packing**

Implement strict `NERResponse(named_entities: list[str])`, `TripleItem(subject: str, predicate: str, object: str)`, `TripleResponse(triples: list[TripleItem])`, and `BatchPhase`, all with `extra="forbid"`. The object form avoids unsupported tuple-generated `prefixItems` in strict Structured Outputs. Render messages with the pinned upstream `PromptTemplateManager` templates `ner` and `triple_extraction`. Each request must have this wire shape:

```python
{
    "custom_id": "ner-musique-" + sha256(
        f"{chunk_id}|{model}|{prompt_sha}|{effective_config_hash}".encode()
    ).hexdigest()[:32],
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": messages,
        "temperature": 0.0,
        "seed": 20260711,
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
}
```

Use `tiktoken.encoding_for_model()` to count the canonical serialized request body, including messages and JSON schema, then add a conservative 64-token per-request overhead. Pack requests in original corpus order until adding one would exceed either configured shard limit. Write JSONL with UTF-8 and a final newline; save a sidecar containing ordered custom IDs, model, full prompt/effective-config hashes, input token upper bound, request SHA-256, phase, dataset, and source-corpus SHA. Resumption rejects any mismatch.

- [ ] **Step 5: Implement submit, poll, and collect without hidden retries**

Use `OpenAI.files.create(file=..., purpose="batch")`, then `OpenAI.batches.create(input_file_id=..., endpoint="/v1/chat/completions", completion_window="24h")`. Persist returned IDs atomically before polling. Poll every 30 seconds until `completed`, `failed`, `expired`, or `cancelled`; never sleep longer than 30 seconds. On completion, download both output and error files. Expose a module `main()` with `prepare`, `submit`, `poll`, and `collect` subcommands so this task does not depend on the unified CLI created later.

`collect_output_rows()` must validate the exact set of expected IDs, require `status_code == 200`, parse `choices[0].message.content`, preserve usage fields, and fail the phase if any row is missing, duplicated, malformed, or has an error. A failed shard may be resubmitted alone with the same request JSONL and a new batch ID; successful shards are immutable.

- [ ] **Step 6: Implement NER-dependent triple requests and upstream export**

Only after every NER shard passes, generate triple requests in corpus order with each passage and its validated entity list. After every triple shard passes, export:

```json
{
  "docs": [
    {
      "idx": "chunk-<md5-of-passage>",
      "passage": "Title\nBody",
      "extracted_entities": ["Entity"],
      "extracted_triples": [["subject", "predicate", "object"]]
    }
  ],
  "avg_ent_chars": 6.0,
  "avg_ent_words": 1.0
}
```

Calculate the upstream chunk ID as `"chunk-" + md5(passage.encode()).hexdigest()`. Filter triples to exactly three nonempty strings and preserve document order. Write to the fingerprinted index directory using the exact filename `openie_results_ner_gpt-4o-mini-2024-07-18.json`.

- [ ] **Step 7: Add the budget preflight**

Before each submission, estimate Batch input/output cost using the frozen pricing snapshot and call `UsageLedger.reserve()` before `files.create`. The upper bound uses the conservative serialized-input count and `ner_max_output_tokens` (512) or `triple_max_output_tokens` (1024) times request count. Pending shards remain active reservations, so submitting several shards cannot bypass the 18 USD ceiling. Batch pricing discount applies only when settled usage came from a completed Batch. The command prints phase, shard, request count, token upper bound, maximum added USD, all pending reservations, and remaining budget.

- [ ] **Step 8: Run fixture tests and one two-document live Batch smoke**

```powershell
uv run pytest tests/test_batch_openie.py -q
$workspace = 'artifacts/smoke/openie'
uv run python -m metagate_hipporag.batch_openie prepare --dataset musique --phase ner --source tests/fixtures/openie_smoke_docs.json --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie submit --dataset musique --phase ner --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie poll --dataset musique --phase ner --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie collect --dataset musique --phase ner --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie prepare --dataset musique --phase triple --source tests/fixtures/openie_smoke_docs.json --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie submit --dataset musique --phase triple --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie poll --dataset musique --phase triple --workspace $workspace --config configs/experiment.yaml
uv run python -m metagate_hipporag.batch_openie collect --dataset musique --phase triple --workspace $workspace --config configs/experiment.yaml
```

`tests/fixtures/openie_smoke_docs.json` contains exactly two synthetic English passages and no benchmark text. Expected: both phases complete, every custom ID appears once, upstream JSON has two documents, cumulative smoke spend remains below 1 USD, and no file is written under `artifacts/indexes/`.

- [ ] **Step 9: Commit**

```powershell
git add configs/experiment.yaml src/metagate_hipporag/batch_openie.py tests/test_batch_openie.py tests/fixtures/batch_*.jsonl tests/fixtures/openie_smoke_docs.json
git commit -m "feat: add resumable Batch OpenIE pipeline"
```

## Task 6: Persist embeddings and fingerprint immutable indexes

**Files:**
- Create: `src/metagate_hipporag/embedding.py`
- Create: `tests/test_embedding.py`
- Modify: `src/metagate_hipporag/provenance.py`

- [ ] **Step 1: Write failing embedding cache tests**

Test that identical `(model, text)` values call the fake backend once, returned arrays are float32 and L2-normalized at the same `batch_encode()` stage as upstream, request usage is recorded only on cache miss, and changing model, dimensions, or instruction mode changes the key. Add explicit cases for `"a\nb" → "a b"` and `"" → " "`, and assert that main experiments use `embedding_instruction_mode="upstream_ignored"` with 3,072 dimensions.

- [ ] **Step 2: Implement `PersistentOpenAIEmbeddingModel`**

Subclass pinned upstream `OpenAIEmbeddingModel`, override both `encode()` and `batch_encode()`, and use a SQLite table:

```sql
CREATE TABLE IF NOT EXISTS embeddings (
  key TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  vector BLOB NOT NULL,
  dimensions INTEGER NOT NULL
)
```

The key is SHA-256 of `{model, dimensions, raw_text, instruction_mode}`. Before the API call, exactly reproduce pinned upstream preprocessing in this order: replace every newline with one space, then replace an empty resulting string with one space. In `upstream_ignored` mode, accept but deliberately do not prepend the upstream instruction; record that mode in every manifest. `encode()` returns unnormalized float32 rows. `batch_encode()` delegates in batches of 64, concatenates in input order, then applies upstream's row-wise normalization only when `embedding_config.norm` is true. Misses call `client.embeddings.create(model=..., input=..., dimensions=3072)` through a ledger reservation; hits use stored float32 bytes. Validate 3,072 dimensions and finite nonzero norms. Never use the upstream bare `except`/`ipdb` path.

- [ ] **Step 3: Implement injection and query-cache export**

```python
def inject_embedding_model(hipporag: object, model: object) -> None:
    hipporag.embedding_model = model
    hipporag.chunk_embedding_store.embedding_model = model
    hipporag.entity_embedding_store.embedding_model = model
    hipporag.fact_embedding_store.embedding_model = model
```

Call this immediately after HippoRAG construction and before `index()` or `retrieve()`. Export `hipporag.query_to_embedding` to `query_embeddings_<effective_config_hash>.npz` at clean shutdown; load it only when dataset, corpus SHA, model, dimensions, instruction mode, upstream SHA, patch SHA, and relevant index-config hash all match.

- [ ] **Step 4: Implement the immutable index fingerprint**

Add `index_config_hash(...)` and `index_directory(...)` to `provenance.py`. The hash payload contains corpus SHA, upstream SHA and package version, compatibility patch SHA, preprocessing version, OpenIE NER/triple prompt hashes, LLM and embedding model IDs, embedding dimensions/instruction mode, and every retrieval/graph-build parameter (`linking_top_k`, PPR damping, passage-node weight, synonym threshold). It deliberately excludes gate, sampling, statistics, and price fields. The directory must be:

```text
artifacts/indexes/<dataset>/<corpus_sha12>/<upstream_sha12>/<llm_slug>/<embedding_slug>/<openie_prompt_sha12>/<index_config_sha12>/
```

If the directory exists with a different full manifest, raise an error. `index_manifest.json` stores every full hash/value above plus document count, OpenIE record count, entity/fact/passages counts, embedding cache row counts, graph file SHA, creation time, GPU/PyTorch information, and `complete: true` only after all invariants pass. Never reuse `force_index_from_scratch` to overwrite mixed state.

- [ ] **Step 5: Run tests and commit**

```powershell
uv run pytest tests/test_embedding.py -q
uv run ruff check src/metagate_hipporag/embedding.py src/metagate_hipporag/provenance.py tests/test_embedding.py
git add src/metagate_hipporag/embedding.py src/metagate_hipporag/provenance.py tests/test_embedding.py
git commit -m "feat: persist embeddings and isolate indexes"
```

## Task 7: Build a traceable HippoRAG bridge and prove baseline equivalence

**Files:**
- Create: `src/metagate_hipporag/hipporag_adapter.py`
- Create: `tests/test_hipporag_adapter.py`
- Create: `tests/test_upstream_contract.py`

- [ ] **Step 1: Write failing bridge unit tests with a fake engine**

Cover graph retrieval, empty-filter dense fallback, chunk-ID mapping, top-k truncation, original query preservation for QA, and capture of facts before/after filtering. The fake engine must expose the same method names as pinned upstream.

- [ ] **Step 2: Implement the narrow bridge**

```python
# src/metagate_hipporag/hipporag_adapter.py (core retrieval path)
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .models import RetrievedPassage, RetrievalTrace, Usage


class HippoRAGBridge:
    def __init__(self, engine: Any, top_k: int = 5) -> None:
        self.engine = engine
        self.top_k = top_k

    def retrieve_with_trace(self, query: str) -> RetrievalTrace:
        started = time.perf_counter()
        if not self.engine.ready_to_retrieve:
            self.engine.prepare_retrieval_objects()
        self.engine.get_query_embeddings([query])
        fact_scores = self.engine.get_fact_scores(query)
        fact_indices, facts, rerank_log = self.engine.rerank_facts(query, fact_scores)
        if facts:
            doc_ids, doc_scores = self.engine.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.engine.global_config.linking_top_k,
                query_fact_scores=fact_scores,
                top_k_facts=facts,
                top_k_fact_indices=fact_indices,
                passage_node_weight=self.engine.global_config.passage_node_weight,
            )
            dense_fallback = False
        else:
            doc_ids, doc_scores = self.engine.dense_passage_retrieval(query)
            dense_fallback = True
        passages: list[RetrievedPassage] = []
        for rank, (doc_id, score) in enumerate(
            zip(doc_ids[: self.top_k], doc_scores[: self.top_k]), start=1
        ):
            chunk_id = self.engine.passage_node_keys[int(doc_id)]
            text = self.engine.chunk_embedding_store.get_row(chunk_id)["content"]
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(score),
                    rank=rank,
                )
            )
        return RetrievalTrace(
            retrieval_query=query,
            passages=passages,
            facts_before_filter=[tuple(value) for value in rerank_log.get("facts_before_rerank", [])],
            facts_after_filter=[tuple(value) for value in rerank_log.get("facts_after_rerank", [])],
            used_dense_fallback=dense_fallback,
            filter_error=rerank_log.get("error"),
            usage=Usage(
                observed_latency_seconds=time.perf_counter() - started,
                method_equivalent_latency_seconds=time.perf_counter() - started,
            ),
        )
```

Add `dense_with_trace()` that calls only `dense_passage_retrieval()`. Add `answer(original_question, passages)` that constructs upstream `QuerySolution(question=original_question, docs=[...], doc_scores=np.array([...]))` and calls `engine.qa()` without another retrieval.

- [ ] **Step 3: Wrap upstream inference for usage snapshots**

Install a wrapper around `engine.llm_model.infer` before constructing the bridge. The wrapper records model, stage, prompt/completion tokens, cache hit, latency, and actual cost; then returns the exact upstream tuple unchanged. Point `engine.rerank_filter.llm_infer_fn` at the same wrapper. Bridge methods take a ledger snapshot before and after a call to assign usage without relying on upstream cumulative timers.

- [ ] **Step 4: Add offline and real-index pinned-upstream contract tests**

For the always-runnable offline test, construct one deterministic engine fixture with NumPy embeddings, five passage rows, candidate facts, and monkeypatched upstream methods. Invoke the pinned unbound `HippoRAG.retrieve(engine, [query], num_to_retrieve=5)[0]` and the bridge on independently reset copies of that fixture. Assert exact document text/chunk-ID order, exact facts-before/after logs, and `np.testing.assert_allclose` on scores with `rtol=1e-7, atol=1e-9`. This test must contain no API response cassette and must fail if any network method is reached.

Also define `assert_real_index_contract(engine, queries)` for Task 11. It snapshots the persistent LLM/embedding caches, runs official and bridge retrieval from the same completed real index, asserts the same order/scores, and verifies zero cache misses during the second path. Task 11 runs it on one fixed development ID per dataset before any complete development run; those local index/cache artifacts remain under `artifacts/` and are not committed.

- [ ] **Step 5: Run checks and commit**

```powershell
uv run pytest tests/test_hipporag_adapter.py -q
uv run pytest tests/test_upstream_contract.py -m integration -q
git add src/metagate_hipporag/hipporag_adapter.py tests/test_hipporag_adapter.py tests/test_upstream_contract.py
git commit -m "feat: expose traceable HippoRAG retrieval"
```

## Task 8: Implement MetaGate, RRF, and threshold selection

**Files:**
- Create: `src/metagate_hipporag/fusion.py`
- Create: `src/metagate_hipporag/metagate.py`
- Create: `tests/test_fusion.py`
- Create: `tests/test_metagate.py`
- Create: `configs/gate_prompt.json`
- Generate in Task 11 after the complete development pass: `configs/gate_threshold.json`

- [ ] **Step 1: Write failing RRF and threshold tests**

```python
# tests/test_fusion.py
from metagate_hipporag.fusion import reciprocal_rank_fusion
from metagate_hipporag.models import RetrievedPassage


def p(chunk_id: str, rank: int) -> RetrievedPassage:
    return RetrievedPassage(chunk_id=chunk_id, text=chunk_id, score=1 / rank, rank=rank)


def test_rrf_deduplicates_by_chunk_id_and_is_deterministic() -> None:
    fused = reciprocal_rank_fusion([[p("a", 1), p("b", 2)], [p("b", 1), p("c", 2)]], k=60, top_k=3)
    assert [row.chunk_id for row in fused] == ["b", "a", "c"]
    assert fused == reciprocal_rank_fusion([[p("a", 1), p("b", 2)], [p("b", 1), p("c", 2)]], k=60, top_k=3)
```

```python
# tests/test_metagate.py
from metagate_hipporag.metagate import select_threshold


def test_threshold_uses_balanced_accuracy_then_lower_expansion() -> None:
    probabilities = [0.9, 0.8, 0.4, 0.2]
    sufficient = [True, True, False, False]
    assert select_threshold(probabilities, sufficient, [0.5, 0.75, 0.85]) == 0.75
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
uv run pytest tests/test_fusion.py tests/test_metagate.py -q
```

Expected: imports fail.

- [ ] **Step 3: Implement deterministic RRF**

```python
# src/metagate_hipporag/fusion.py
from __future__ import annotations

from collections import defaultdict

from .models import RetrievedPassage


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedPassage]], k: int, top_k: int
) -> list[RetrievedPassage]:
    scores: dict[str, float] = defaultdict(float)
    passage_by_id: dict[str, RetrievedPassage] = {}
    for ranking in rankings:
        for passage in ranking:
            scores[passage.chunk_id] += 1.0 / (k + passage.rank)
            passage_by_id.setdefault(passage.chunk_id, passage)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return [
        RetrievedPassage(
            chunk_id=chunk_id,
            text=passage_by_id[chunk_id].text,
            score=scores[chunk_id],
            rank=rank,
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
```

- [ ] **Step 4: Freeze the exact zero-shot gate prompt**

Use this system prompt verbatim and store its SHA-256 in `configs/gate_prompt.json`:

```text
You are an evidence-sufficiency monitor for retrieval-augmented question answering.
Judge only whether the supplied passages contain all information needed to answer the
question. Do not answer the question. A passage set is sufficient only when every
required bridge in a multi-hop question is supported. Ignore your parametric knowledge.
Return: (1) a probability from 0 to 1 that the evidence is sufficient; (2) the most
important missing information, or "none"; (3) a concise standalone retrieval query that
would find the missing evidence, and that remains useful when evidence is already
sufficient; and (4) a factual rationale summary of at most 40 words. Never reveal a
private chain of thought.
```

The first-gate user message serializes the original question, `facts_before_filter`, `facts_after_filter`, and five passages with stable numeric labels. The second-gate message serializes the original question, both retrieval queries, each round's before/after fact logs, and the five fused passages. Neither message may include dataset name, gold answer, gold documents, stratum, recall, method name, or whether expansion was mandatory.

- [ ] **Step 5: Implement gate calls and threshold selection**

`MetaGate.decide()` calls `CachedStructuredClient.complete(..., response_model=GateDecision)`. `select_threshold()` evaluates candidates on the three equal-sized pooled development sets where the target is `Recall@5 == 1`, maximizes `sklearn.metrics.balanced_accuracy_score`, then chooses lower expansion rate and finally the higher threshold. The function accepts a `split_name` literal `"dev"` and rejects every other value. Task 11 persists the chosen threshold with effective config hash, prompt hash, exact development IDs, target counts by dataset, candidate scores, chosen score, and timestamp; loading refuses any hash mismatch. The same threshold may flag second-round low confidence, but second-round calibration/selective-risk findings remain exploratory because that gate sees a selected evidence distribution.

- [ ] **Step 6: Implement the bounded policy**

For MetaGate: first bridge retrieval → first gate → stop when probability ≥ threshold; otherwise retrieve once using `retrieval_rewrite`, fuse the two rankings, run a second gate, always produce a forced answer, and set `abstain_flag` when second probability < threshold. For Always-Expand: reuse the same first gate decision, always perform the second retrieval and identical RRF, and run the identical second gate; it records confidence but never uses either probability to skip retrieval or forced answering. `max_expansions` must equal 1 at validation time.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run pytest tests/test_fusion.py tests/test_metagate.py -q
uv run ruff check src/metagate_hipporag/fusion.py src/metagate_hipporag/metagate.py tests/test_fusion.py tests/test_metagate.py
git add src/metagate_hipporag/fusion.py src/metagate_hipporag/metagate.py tests/test_fusion.py tests/test_metagate.py configs/gate_prompt.json
git commit -m "feat: add metacognitive retrieval gate"
```
