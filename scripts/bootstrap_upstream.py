from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "third_party" / "HippoRAG"
PATCH_FILE = REPO_ROOT / "patches" / "hipporag-openai-only.patch"
PATCH_HASH_FILE = REPO_ROOT / "patches" / "hipporag-openai-only.patch.sha256"
UPSTREAM_URL = "https://github.com/OSU-NLP-Group/HippoRAG.git"
UPSTREAM_COMMIT = "ad30fc3e2062202d9e975e32cd28212424a56ccb"
UPSTREAM_PACKAGE_VERSION = "2.0.0-alpha.4"
PATCH_MARKER = UPSTREAM_DIR / ".metagate-openai-only-patch"

_EXPECTED_PATCH_PATHS = frozenset(
    {
        "src/hipporag/embedding_model/__init__.py",
        "src/hipporag/llm/__init__.py",
    }
)


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_patch_hash() -> str:
    if not PATCH_FILE.exists():
        raise RuntimeError(f"patch file not found: {PATCH_FILE}")
    actual = sha256_file(PATCH_FILE)
    if PATCH_HASH_FILE.exists():
        expected = PATCH_HASH_FILE.read_text(encoding="utf-8").strip()
        if actual != expected:
            raise RuntimeError(f"patch hash mismatch: expected {expected}, got {actual}")
    return actual


def verify_patch_state() -> None:
    """Verify that the applied patch touches only the two expected files."""
    changed = run("git", "diff", "--name-only", cwd=UPSTREAM_DIR)
    changed_paths = {p.strip() for p in changed.split("\n") if p.strip()}
    if changed_paths != _EXPECTED_PATCH_PATHS:
        raise RuntimeError(
            f"patch affected unexpected files: {sorted(changed_paths)}; "
            f"expected exactly {sorted(_EXPECTED_PATCH_PATHS)}"
        )
    run("git", "diff", "--check", cwd=UPSTREAM_DIR)


def verify_checkout(path: Path, actual_commit: str | None = None) -> None:
    if not (path / ".git").exists():
        raise RuntimeError(f"missing git checkout: {path}")
    actual = actual_commit or run("git", "rev-parse", "HEAD", cwd=path)
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"unexpected HippoRAG commit: {actual}")


def verify_package_version() -> None:
    setup_py = UPSTREAM_DIR / "setup.py"
    if not setup_py.exists():
        raise RuntimeError(f"setup.py not found in {UPSTREAM_DIR}")
    content = setup_py.read_text(encoding="utf-8")
    # Look for version string like version="2.0.0-alpha.4"
    for line in content.split("\n"):
        if "version" in line and UPSTREAM_PACKAGE_VERSION in line:
            return
    raise RuntimeError(
        f"expected package version {UPSTREAM_PACKAGE_VERSION} not found in setup.py"
    )


def write_patch_marker(patch_hash: str) -> None:
    marker = {
        "upstream_commit": UPSTREAM_COMMIT,
        "patch_sha256": patch_hash,
        "package_version": UPSTREAM_PACKAGE_VERSION,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    PATCH_MARKER.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def bootstrap(check_only: bool) -> None:
    if not UPSTREAM_DIR.exists():
        if check_only:
            raise RuntimeError(f"missing upstream checkout: {UPSTREAM_DIR}")
        UPSTREAM_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning from {UPSTREAM_URL} ...")
        run("git", "clone", UPSTREAM_URL, str(UPSTREAM_DIR))
        run("git", "checkout", "--detach", UPSTREAM_COMMIT, cwd=UPSTREAM_DIR)

    verify_checkout(UPSTREAM_DIR)

    patch_hash = verify_patch_hash()

    if not PATCH_MARKER.exists():
        if check_only:
            raise RuntimeError("compatibility patch has not been applied")
        run("git", "apply", "--check", str(PATCH_FILE), cwd=UPSTREAM_DIR)
        run("git", "apply", str(PATCH_FILE), cwd=UPSTREAM_DIR)
        verify_patch_state()
        verify_package_version()
        write_patch_marker(patch_hash)

    verify_checkout(UPSTREAM_DIR)

    # Re-verify patch state on every invocation
    verify_patch_state()
    verify_package_version()

    if not check_only:
        run(
            "uv", "pip", "install", "--no-deps", "--editable", str(UPSTREAM_DIR),
            cwd=REPO_ROOT,
        )

    verify_checkout(UPSTREAM_DIR)

    # Write/update patch hash file
    if not PATCH_HASH_FILE.exists() or PATCH_HASH_FILE.read_text(encoding="utf-8").strip() != patch_hash:
        PATCH_HASH_FILE.write_text(patch_hash + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bootstrap(check_only=args.check)
    print(
        f"HippoRAG ready at {UPSTREAM_COMMIT} "
        f"(version {UPSTREAM_PACKAGE_VERSION}, "
        f"patch {sha256_file(PATCH_FILE)[:12]})"
    )


if __name__ == "__main__":
    sys.exit(main())
