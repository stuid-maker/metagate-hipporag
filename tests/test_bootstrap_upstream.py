# tests/test_bootstrap_upstream.py
from pathlib import Path

import pytest

from scripts.bootstrap_upstream import (
    PATCH_FILE,
    UPSTREAM_COMMIT,
    UPSTREAM_PACKAGE_VERSION,
    verify_checkout,
)


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
    with pytest.raises(RuntimeError, match="unexpected HippoRAG commit"):
        verify_checkout(tmp_path, actual_commit="0" * 40)
