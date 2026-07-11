"""SHA-256, atomic files, JSONL, and SQLite-backed UsageLedger."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from .models import LedgerEntry, LedgerSnapshot, Usage

# ── Hashing ─────────────────────────────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


# ── Atomic file I/O ─────────────────────────────────────────────────────────


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


def read_jsonl_recover_tail(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, recovering from a corrupt (incomplete) final line.

    If the last line is a non-empty fragment that is invalid JSON, the file is
    backed up to ``<name>.corrupt.<UTC timestamp>``, the corrupt tail is
    truncated, and the valid records are returned.

    Any invalid JSON in an earlier line is a hard error.
    """
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8")

    if not raw:
        return []

    lines = raw.split("\n")
    # Trailing empty string from final newline
    if lines and lines[-1] == "":
        lines.pop()

    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(lines) - 1:
                # Last line is corrupt → recover
                _backup_and_truncate(path, lines, idx)
                return rows
            # Earlier line is corrupt → hard error
            raise ValueError(
                f"corrupt JSON at line {idx + 1} in {path}: {line[:200]!r}"
            ) from None

    return rows


def _backup_and_truncate(
    path: Path, lines: list[str], corrupt_idx: int
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(f"{path.name}.corrupt.{ts}")
    lock = FileLock(str(path) + ".lock")
    with lock:
        path.replace(backup)
        valid = "\n".join(lines[:corrupt_idx])
        if valid:
            valid += "\n"
        path.write_text(valid, encoding="utf-8")


# ── UsageLedger ─────────────────────────────────────────────────────────────


def _serialize_entry(entry: LedgerEntry) -> str:
    return entry.model_dump_json()


def _deserialize_entry(raw: str) -> LedgerEntry:
    return LedgerEntry.model_validate_json(raw)


def _serialize_reservation(
    reservation_id: str, upper_bound_actual_usd: float, state: str
) -> str:
    return json.dumps(
        {
            "reservation_id": reservation_id,
            "upper_bound_actual_usd": upper_bound_actual_usd,
            "state": state,
        },
        sort_keys=True,
    )


class UsageLedger:
    """SQLite-backed cost ledger with budget enforcement."""

    def __init__(self, db_path: Path, limit_usd: float) -> None:
        self._db_path = db_path
        self._limit_usd = limit_usd
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ledger_entries (
                    event_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    upper_bound_actual_usd REAL NOT NULL,
                    state TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )"""
            )
            conn.execute(
                "INSERT OR IGNORE INTO state(key, value) VALUES('limit_usd', ?)",
                (str(self._limit_usd),),
            )

    def _current_actual_cost(self, conn: sqlite3.Connection) -> float:
        row = conn.execute(
            "SELECT SUM(json_extract(payload, '$.actual_usd')) FROM ledger_entries"
        ).fetchone()
        return row[0] if row[0] is not None else 0.0

    def _active_reservation_sum(self, conn: sqlite3.Connection) -> float:
        row = conn.execute(
            "SELECT SUM(upper_bound_actual_usd) FROM reservations "
            "WHERE state = 'active'"
        ).fetchone()
        return row[0] if row[0] is not None else 0.0

    def _check_budget(
        self, conn: sqlite3.Connection, upper_bound: float
    ) -> None:
        limit_row = conn.execute(
            "SELECT value FROM state WHERE key = 'limit_usd'"
        ).fetchone()
        limit = float(limit_row[0]) if limit_row else self._limit_usd
        current = self._current_actual_cost(conn)
        active = self._active_reservation_sum(conn)
        if current + active + upper_bound > limit:
            raise ValueError(
                f"budget exceeded: settled={current:.4f} + active_reservations="
                f"{active:.4f} + new_upper_bound={upper_bound:.4f} "
                f"> limit={limit:.4f}"
            )

    def _next_sequence(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT MAX(id) FROM snapshots").fetchone()
        return (row[0] or 0) + 1

    # ── Public API ──────────────────────────────────────────────────────

    def reserve(
        self, reservation_id: str, upper_bound_actual_usd: float, limit_usd: float
    ) -> None:
        """Reserve budget for an upcoming operation."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._check_budget(conn, upper_bound_actual_usd)
                payload = _serialize_reservation(
                    reservation_id, upper_bound_actual_usd, "active"
                )
                conn.execute(
                    "INSERT INTO reservations"
                    "(reservation_id, upper_bound_actual_usd, state, payload) "
                    "VALUES (?, ?, 'active', ?)",
                    (reservation_id, upper_bound_actual_usd, payload),
                )
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def append(self, entry: LedgerEntry) -> None:
        """Append a single entry (idempotent if byte-identical)."""
        with self._connect() as conn:
            payload = _serialize_entry(entry)
            existing = conn.execute(
                "SELECT payload FROM ledger_entries WHERE event_id = ?",
                (entry.event_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise ValueError(
                        f"conflicting duplicate event_id {entry.event_id}: "
                        f"existing payload differs"
                    )
                return  # idempotent — same bytes
            conn.execute(
                "INSERT INTO ledger_entries(event_id, payload) VALUES (?, ?)",
                (entry.event_id, payload),
            )
            conn.commit()

    def settle(
        self, reservation_id: str, entries: list[LedgerEntry]
    ) -> None:
        """Atomically append entries and close the reservation."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state FROM reservations WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"unknown reservation: {reservation_id}"
                    )
                if row[0] != "active":
                    raise ValueError(
                        f"reservation {reservation_id} is not active (state={row[0]})"
                    )

                for entry in entries:
                    payload = _serialize_entry(entry)
                    existing = conn.execute(
                        "SELECT payload FROM ledger_entries WHERE event_id = ?",
                        (entry.event_id,),
                    ).fetchone()
                    if existing is not None:
                        if existing[0] != payload:
                            raise ValueError(
                                f"conflicting duplicate event_id {entry.event_id}"
                            )
                        continue
                    conn.execute(
                        "INSERT INTO ledger_entries(event_id, payload) VALUES (?, ?)",
                        (entry.event_id, payload),
                    )

                conn.execute(
                    "UPDATE reservations SET state = 'settled' WHERE reservation_id = ?",
                    (reservation_id,),
                )
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

        # Record a snapshot after settling
        self._record_snapshot()

    def release(self, reservation_id: str, reason: str) -> None:
        """Release an active reservation without recording entries."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state FROM reservations WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"unknown reservation: {reservation_id}"
                    )
                new_state = f"released:{reason}"
                conn.execute(
                    "UPDATE reservations SET state = ? WHERE reservation_id = ?",
                    (new_state, reservation_id),
                )
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def snapshot(self) -> LedgerSnapshot:
        """Return the current cumulative snapshot."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT
                    COALESCE(SUM(json_extract(payload, '$.actual_usd')), 0),
                    COALESCE(SUM(json_extract(payload, '$.method_equivalent_usd')), 0),
                    COALESCE(SUM(json_extract(payload, '$.prompt_tokens')), 0),
                    COALESCE(SUM(json_extract(payload, '$.completion_tokens')), 0),
                    COALESCE(SUM(json_extract(payload, '$.embedding_tokens')), 0),
                    COALESCE(SUM(json_extract(payload, '$.observed_latency_seconds')), 0),
                    COALESCE(SUM(json_extract(payload, '$.method_equivalent_latency_seconds')), 0)
                FROM ledger_entries"""
            ).fetchone()

        seq_row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM snapshots").fetchone()
        seq = seq_row[0] if seq_row else 0

        return LedgerSnapshot(
            sequence=seq,
            actual_usd=row[0],
            method_equivalent_usd=row[1],
            prompt_tokens=row[2],
            completion_tokens=row[3],
            embedding_tokens=row[4],
            observed_latency_seconds=row[5],
            method_equivalent_latency_seconds=row[6],
        )

    def delta(self, before: LedgerSnapshot, after: LedgerSnapshot) -> Usage:
        """Compute the incremental usage between two snapshots."""
        return Usage(
            actual_usd=max(0.0, after.actual_usd - before.actual_usd),
            method_equivalent_usd=max(
                0.0, after.method_equivalent_usd - before.method_equivalent_usd
            ),
            prompt_tokens=max(0, after.prompt_tokens - before.prompt_tokens),
            completion_tokens=max(
                0, after.completion_tokens - before.completion_tokens
            ),
            embedding_tokens=max(
                0, after.embedding_tokens - before.embedding_tokens
            ),
            observed_latency_seconds=max(
                0.0,
                after.observed_latency_seconds - before.observed_latency_seconds,
            ),
            method_equivalent_latency_seconds=max(
                0.0,
                after.method_equivalent_latency_seconds
                - before.method_equivalent_latency_seconds,
            ),
        )

    def _record_snapshot(self) -> None:
        """Internal: persist the current snapshot as a numbered row."""
        snap = self.snapshot()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots(payload) VALUES (?)",
                (snap.model_dump_json(),),
            )
            conn.commit()
