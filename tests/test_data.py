"""Tests for data normalization, download, and deterministic splits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from metagate_hipporag.data import (
    DATASET_FILES,
    HF_REVISION,
    deterministic_split,
    normalize_example,
    upstream_gold_docs,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ── normalization ────────────────────────────────────────────────────────────


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


def test_nq_rear_normalization() -> None:
    raw = {
        "question": "test question?",
        "reference": ["Answer A", "Answer B"],
        "contexts": [
            {"title": "Gold Title", "text": "Gold text.", "is_supporting": True},
            {"title": "Noise", "text": "Noise text.", "is_supporting": False},
        ],
    }
    example = normalize_example("nq_rear", raw, fallback_id="nq_0")
    assert example.example_id == "nq_0"
    assert example.gold_answers == ["Answer A", "Answer B"]
    assert example.gold_docs == ["Gold Title\nGold text."]
    assert example.stratum == "simple"


def test_2wiki_normalization() -> None:
    raw = {
        "_id": "abc123",
        "type": "compositional",
        "question": "When did X die?",
        "answer": "2020",
        "supporting_facts": [["Doc A", 0]],
        "context": [
            ["Doc A", ["Doc A sentence 0 says X died in 2020.", "More detail."]],
            ["Noise", ["Not relevant."]],
        ],
    }
    example = normalize_example("2wikimultihopqa", raw)
    assert example.example_id == "abc123"
    assert example.gold_answers == ["2020"]
    # Upstream joins ALL sentences for the supporting title, not just indexed ones
    assert example.gold_docs == [
        "Doc A\nDoc A sentence 0 says X died in 2020. More detail."
    ]
    assert example.stratum == "compositional"


def test_2wiki_answer_list() -> None:
    """2Wiki answer can be a list."""
    raw = {
        "_id": "multi",
        "type": "comparison",
        "question": "Compare?",
        "answer": ["A", "B"],
        "supporting_facts": [],
        "context": [],
    }
    example = normalize_example("2wikimultihopqa", raw)
    assert example.gold_answers == ["A", "B"]


def test_nq_rear_id_is_required() -> None:
    """nq_rear has no id/_id field — fallback_id must be provided."""
    raw = {"question": "q?", "reference": ["A"], "contexts": []}
    with pytest.raises(ValueError, match="fallback_id"):
        normalize_example("nq_rear", raw)


# ── upstream_gold_docs ──────────────────────────────────────────────────────


def test_gold_docs_nq_rear() -> None:
    raw = {
        "question": "q?",
        "reference": ["A"],
        "contexts": [
            {"title": "T1", "text": "Body 1", "is_supporting": True},
            {"title": "T2", "text": "Body 2", "is_supporting": False},
        ],
    }
    docs = upstream_gold_docs("nq_rear", raw)
    assert docs == ["T1\nBody 1"]


def test_gold_docs_musique() -> None:
    raw = {
        "paragraphs": [
            {"title": "Gold", "paragraph_text": "Ev", "is_supporting": True},
            {"title": "Gold2", "text": "Ev2", "is_supporting": True},
            {"title": "Noise", "paragraph_text": "No", "is_supporting": False},
        ]
    }
    docs = upstream_gold_docs("musique", raw)
    assert docs == ["Gold\nEv", "Gold2\nEv2"]


def test_gold_docs_2wiki() -> None:
    raw = {
        "supporting_facts": [["T1", 0], ["T1", 2]],
        "context": [
            ["T1", ["s0", "s1", "s2"]],
            ["T2", ["not used"]],
        ],
    }
    docs = upstream_gold_docs("2wikimultihopqa", raw)
    # Upstream joins ALL sentences of a supporting title, not just individual indices
    assert docs == ["T1\ns0 s1 s2"]


def test_gold_docs_unknown_schema_raises() -> None:
    with pytest.raises(ValueError, match="unsupported gold-document schema"):
        upstream_gold_docs("nq_rear", {})


# ── deterministic split ─────────────────────────────────────────────────────


def test_split_is_disjoint_reproducible_and_stratified() -> None:
    rows = [
        normalize_example(
            "musique",
            {
                "id": f"{2 + i % 3}hop__{i}",
                "question": f"q{i}",
                "answer": f"a{i}",
                "paragraphs": [
                    {
                        "title": f"g{i}-{j}",
                        "paragraph_text": "e",
                        "is_supporting": True,
                    }
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
    strata = Counter(
        row.stratum for row in rows if row.example_id in first["test_ids"]
    )
    assert set(strata) == {"2", "3", "4"}


def test_split_rejects_duplicate_ids() -> None:
    rows = [
        normalize_example("musique", {"id": "dup", "question": "q", "answer": "a", "paragraphs": []}),
        normalize_example("musique", {"id": "dup", "question": "q2", "answer": "a2", "paragraphs": []}),
    ]
    with pytest.raises(ValueError, match="duplicate example IDs"):
        deterministic_split(rows, dev_size=0, test_size=0, seed=1, source_sha256="0" * 64)


def test_split_size_exceeds_population() -> None:
    # Create rows with all three required strata (2-hop, 3-hop, 4-hop)
    rows: list[Example] = []
    for i in range(2):
        rows.append(
            normalize_example(
                "musique",
                {
                    "id": f"a{i}",
                    "question": "q",
                    "answer": "a",
                    "paragraphs": [
                        {"title": f"T{j}", "paragraph_text": "e", "is_supporting": True}
                        for j in range(2)  # 2 supporting → stratum "2"
                    ],
                },
            )
        )
    for i in range(2):
        rows.append(
            normalize_example(
                "musique",
                {
                    "id": f"b{i}",
                    "question": "q",
                    "answer": "a",
                    "paragraphs": [
                        {"title": f"T{j}", "paragraph_text": "e", "is_supporting": True}
                        for j in range(3)  # 3 supporting → stratum "3"
                    ],
                },
            )
        )
    for i in range(2):
        rows.append(
            normalize_example(
                "musique",
                {
                    "id": f"c{i}",
                    "question": "q",
                    "answer": "a",
                    "paragraphs": [
                        {"title": f"T{j}", "paragraph_text": "e", "is_supporting": True}
                        for j in range(4)  # 4 supporting → stratum "4"
                    ],
                },
            )
        )
    # 6 total rows, request 4+4=8 → exceeds population
    with pytest.raises(ValueError, match="exceed population"):
        deterministic_split(rows, dev_size=4, test_size=4, seed=1, source_sha256="0" * 64)


# ── fixtures ────────────────────────────────────────────────────────────────


def test_musique_fixture_roundtrip() -> None:
    data = json.loads((FIXTURES / "musique_small.json").read_text(encoding="utf-8"))
    assert len(data) == 3
    for raw in data:
        example = normalize_example("musique", raw)
        assert example.dataset == "musique"
        assert len(example.gold_answers) > 0
        assert len(example.gold_docs) > 0
        assert example.stratum in {"1", "2", "3", "4"}


def test_nq_fixture_roundtrip() -> None:
    data = json.loads((FIXTURES / "nq_small.json").read_text(encoding="utf-8"))
    assert len(data) == 2
    for i, raw in enumerate(data):
        example = normalize_example("nq_rear", raw, fallback_id=f"nq_{i}")
        assert example.dataset == "nq_rear"
        assert len(example.gold_answers) > 0
        assert example.stratum == "simple"


def test_2wiki_fixture_roundtrip() -> None:
    data = json.loads((FIXTURES / "twowiki_small.json").read_text(encoding="utf-8"))
    assert len(data) == 3
    for raw in data:
        example = normalize_example("2wikimultihopqa", raw)
        assert example.dataset == "2wikimultihopqa"
        assert len(example.gold_answers) > 0
        assert example.stratum in {"compositional", "comparison", "inference"}


# ── constants ───────────────────────────────────────────────────────────────


def test_hf_revision_is_frozen() -> None:
    assert HF_REVISION == "5ec05b38deecc3318bb432c69865959c56058990"
    assert len(HF_REVISION) == 40


def test_six_files_are_expected() -> None:
    names: list[str] = []
    for pair in DATASET_FILES.values():
        names.extend(pair)
    assert len(names) == 6
    for name in names:
        assert name.endswith(".json")
