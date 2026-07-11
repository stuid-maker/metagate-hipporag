"""Pinned downloads, schema adapters, gold normalization, and fixed stratified splits."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import load_config
from .models import DatasetId, Example
from .provenance import atomic_write_json, sha256_file

HF_REVISION = "5ec05b38deecc3318bb432c69865959c56058990"
HF_BASE = (
    f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{HF_REVISION}"
)
DATASET_FILES: dict[str, tuple[str, str]] = {
    dataset: (f"{dataset}.json", f"{dataset}_corpus.json")
    for dataset in ("nq_rear", "musique", "2wikimultihopqa")
}

# Expected corpus counts per dataset
EXPECTED_CORPUS_COUNTS: dict[str, int] = {
    "nq_rear": 9633,
    "musique": 11656,
    "2wikimultihopqa": 6119,
}
EXPECTED_QUESTION_COUNT = 1000


# ── document helpers ─────────────────────────────────────────────────────────


def document_text(title: str, text: str) -> str:
    return f"{title}\n{text}"


# ── gold-document extraction (matches upstream main.py::get_gold_docs) ──────


def upstream_gold_docs(dataset: DatasetId, raw: dict[str, Any]) -> list[str]:
    """Extract gold documents using the upstream HippoRAG logic."""
    if "supporting_facts" in raw:
        # 2WikiMultiHopQA format
        supporting_titles = {title for title, _ in raw["supporting_facts"]}
        # Build a map from title to ordered sentence list
        title_sentences: dict[str, list[str]] = {}
        for title, sentences in raw["context"]:
            title_sentences[title] = sentences
        gold = [
            document_text(title, " ".join(sentences))
            for title, sentences in title_sentences.items()
            if title in supporting_titles
        ]
    elif "contexts" in raw:
        # NQ-REaR format
        gold = [
            document_text(row["title"], row["text"])
            for row in raw["contexts"]
            if row.get("is_supporting", True)
        ]
    elif "paragraphs" in raw:
        # MuSiQue format — text field may be "text" or "paragraph_text"
        supporting = [
            row for row in raw["paragraphs"] if row.get("is_supporting", True)
        ]
        gold = [
            document_text(
                row["title"],
                row.get("text") or row.get("paragraph_text", ""),
            )
            for row in supporting
        ]
    else:
        raise ValueError(
            f"unsupported gold-document schema for {dataset}: "
            f"expected supporting_facts, contexts, or paragraphs; "
            f"got keys {sorted(raw.keys())!r}"
        )
    # Deduplicate while preserving order
    return list(dict.fromkeys(gold))


# ── normalization ────────────────────────────────────────────────────────────


def normalize_example(
    dataset: DatasetId,
    raw: dict[str, Any],
    *,
    fallback_id: str | None = None,
) -> Example:
    """Normalize a raw dataset record into an Example.

    NQ-REaR has no ``id`` or ``_id`` field — pass ``fallback_id``.
    """
    gold = upstream_gold_docs(dataset, raw)

    # ── answer extraction ────────────────────────────────────────────
    if "answer" in raw:
        answer_value = raw["answer"]
    elif "reference" in raw:
        answer_value = raw["reference"]  # NQ-REaR
    else:
        raise ValueError(
            f"no answer/reference field in {dataset} record: "
            f"keys={sorted(raw.keys())!r}"
        )

    answers: list[str] = (
        answer_value if isinstance(answer_value, list) else [str(answer_value)]
    )
    # Append aliases when available (MuSiQue)
    aliases = raw.get("answer_aliases")
    if isinstance(aliases, list):
        answers.extend(aliases)

    # ── example_id ───────────────────────────────────────────────────
    if "id" in raw:
        example_id = raw["id"]
    elif "_id" in raw:
        example_id = raw["_id"]
    elif fallback_id is not None:
        example_id = fallback_id
    else:
        raise ValueError(
            f"no id/_id field in {dataset} record and no fallback_id: "
            f"keys={sorted(raw.keys())!r}"
        )

    # ── stratum ──────────────────────────────────────────────────────
    if dataset == "musique":
        stratum = str(len(gold))
    elif dataset == "2wikimultihopqa":
        stratum = raw.get("type", "unknown")
    else:
        stratum = "simple"

    return Example(
        dataset=dataset,
        example_id=str(example_id),
        question=raw["question"],
        gold_answers=list(dict.fromkeys(answers)),  # deduplicate, preserve order
        gold_docs=gold,
        stratum=stratum,
    )


# ── download ─────────────────────────────────────────────────────────────────


def _download_file(url: str, dest: Path, *, timeout: float = 120.0) -> dict[str, str]:
    """Stream-download a file with SHA-256 verification and atomic rename.

    Returns a manifest record for this file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    digest = hashlib.sha256()

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)

    file_hash = digest.hexdigest()
    file_size = tmp.stat().st_size

    if file_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded empty file: {url}")

    # Atomic rename
    os.replace(tmp, dest)

    return {
        "url": url,
        "file": str(dest.relative_to(dest.parents[2])),  # relative to repo root
        "size_bytes": file_size,
        "sha256": file_hash,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_data(data_dir: Path) -> dict[str, Any]:
    """Download all 6 dataset files and return a manifest dict.

    Skips files that already exist with a matching SHA-256.
    Does NOT re-download on mismatch — raises instead.
    """
    manifest_path = data_dir / "manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    records: list[dict[str, str]] = []
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for dataset, (questions_file, corpus_file) in DATASET_FILES.items():
        for filename in (questions_file, corpus_file):
            url = f"{HF_BASE}/{filename}"
            dest = raw_dir / filename

            # Check existing file
            if dest.exists():
                actual_hash = sha256_file(dest)
                # Look up expected hash from existing manifest
                expected_hash = None
                for rec in existing_manifest.get("files", []):
                    if rec.get("file", "").endswith(filename):
                        expected_hash = rec.get("sha256")
                        break
                if expected_hash and actual_hash != expected_hash:
                    raise RuntimeError(
                        f"hash mismatch for {filename}: "
                        f"expected {expected_hash}, got {actual_hash}"
                    )
                # File exists and is valid — reuse
                records.append({
                    "url": url,
                    "file": f"data/raw/{filename}",
                    "size_bytes": dest.stat().st_size,
                    "sha256": actual_hash,
                    "downloaded_at_utc": existing_manifest.get(
                        "generated_at_utc", ""
                    ),
                })
                continue

            print(f"Downloading {filename} …", file=sys.stderr)
            rec = _download_file(url, dest)
            records.append(rec)

    manifest = {
        "revision": HF_REVISION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


# ── stratified split ─────────────────────────────────────────────────────────


def _allocate(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation with remainder distributed to largest remainders."""
    population = sum(group_sizes.values())
    if total < 0 or total > population or population == 0:
        raise ValueError(
            f"requested allocation {total} exceeds population {population}"
        )
    raw = {name: total * size / population for name, size in group_sizes.items()}
    allocated = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(allocated.values())
    # Distribute remainder to strata with largest fractional remainders
    order = sorted(
        raw, key=lambda name: (-(raw[name] - allocated[name]), name)
    )
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
    """Create a deterministic, stratified, non-overlapping dev/test split.

    Returns a dict suitable for JSON serialisation.
    """
    if not rows or len(source_sha256) != 64:
        raise ValueError("rows and full source SHA-256 are required")

    datasets = {row.dataset for row in rows}
    if len(datasets) != 1:
        raise ValueError("a split may contain exactly one dataset")
    dataset = next(iter(datasets))

    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("duplicate example IDs")

    # Group by stratum
    groups: dict[str, list[Example]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item.example_id):
        groups[row.stratum].append(row)

    # Validate expected strata
    expected_strata: set[str] = {
        "nq_rear": {"simple"},
        "musique": {"2", "3", "4"},
        "2wikimultihopqa": {
            "compositional",
            "comparison",
            "bridge_comparison",
            "inference",
        },
    }[dataset]
    if set(groups) != expected_strata:
        raise ValueError(
            f"unexpected {dataset} strata: got {sorted(groups)!r}, "
            f"expected {sorted(expected_strata)!r}"
        )

    if dev_size + test_size > len(rows):
        raise ValueError(
            f"development ({dev_size}) and test ({test_size}) sizes "
            f"exceed population ({len(rows)})"
        )

    # Proportional allocation
    dev_alloc = _allocate(
        {key: len(value) for key, value in groups.items()}, dev_size
    )
    remaining_sizes = {
        key: len(value) - dev_alloc[key] for key, value in groups.items()
    }
    test_alloc = _allocate(remaining_sizes, test_size)

    # Shuffle deterministically per stratum
    rng = random.Random(seed)
    dev_ids: list[str] = []
    test_ids: list[str] = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        dev_count = dev_alloc[key]
        test_count = test_alloc[key]
        dev_ids.extend(row.example_id for row in group[:dev_count])
        test_ids.extend(
            row.example_id
            for row in group[dev_count : dev_count + test_count]
        )

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


# ── CLI ──────────────────────────────────────────────────────────────────────


def prepare(config_path: str) -> None:
    """Download data and freeze splits (``prepare`` command)."""
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(Path(config_path))

    # 1. Download
    print("=== Downloading data ===", file=sys.stderr)
    manifest = download_data(repo_root / "data")
    n_files = len(manifest.get("files", []))
    print(f"  {n_files} files ready", file=sys.stderr)

    # 2. Load and normalise
    print("=== Normalising examples ===", file=sys.stderr)
    raw_dir = repo_root / "data" / "raw"
    splits_dir = repo_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id in ("nq_rear", "musique", "2wikimultihopqa"):
        questions_file = raw_dir / f"{dataset_id}.json"
        if not questions_file.exists():
            raise RuntimeError(
                f"missing {questions_file} — run download first"
            )

        raw_data = json.loads(questions_file.read_text(encoding="utf-8"))
        if len(raw_data) != EXPECTED_QUESTION_COUNT:
            raise RuntimeError(
                f"{dataset_id}: expected {EXPECTED_QUESTION_COUNT} questions, "
                f"got {len(raw_data)}"
            )

        # Normalise
        examples: list[Example] = []
        for i, raw in enumerate(raw_data):
            fallback_id = (
                f"{dataset_id}_{i}" if dataset_id == "nq_rear" else None
            )
            examples.append(
                normalize_example(dataset_id, raw, fallback_id=fallback_id)
            )

        # Verify corpus count
        corpus_file = raw_dir / f"{dataset_id}_corpus.json"
        if corpus_file.exists():
            corpus_data = json.loads(corpus_file.read_text(encoding="utf-8"))
            expected_corpus = EXPECTED_CORPUS_COUNTS.get(dataset_id)
            if expected_corpus is not None and len(corpus_data) != expected_corpus:
                print(
                    f"  WARNING: {dataset_id}_corpus.json has {len(corpus_data)} "
                    f"entries, expected {expected_corpus}",
                    file=sys.stderr,
                )

        # Compute source SHA-256
        source_hash = sha256_file(questions_file)

        # Create split
        split = deterministic_split(
            examples,
            dev_size=config.sampling.dev_per_dataset,
            test_size=config.sampling.test_per_dataset,
            seed=config.project.seed,
            source_sha256=source_hash,
        )

        split_path = splits_dir / f"{dataset_id}.json"
        atomic_write_json(split_path, split)
        print(
            f"  {dataset_id}: {len(split['dev_ids'])} dev / "
            f"{len(split['test_ids'])} test  "
            f"(strata: { {k: v['population'] for k, v in split['stratum_counts'].items()} })",
            file=sys.stderr,
        )

    print("=== Done ===", file=sys.stderr)


def main() -> None:
    """Entry point for ``python -m metagate_hipporag.data``."""
    if len(sys.argv) < 2:
        print("usage: python -m metagate_hipporag.data prepare --config <path>", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    if command == "prepare":
        # Parse --config
        if "--config" in sys.argv:
            idx = sys.argv.index("--config")
            config_path = sys.argv[idx + 1]
        else:
            config_path = "configs/experiment.yaml"
        prepare(config_path)
    else:
        print(f"unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
