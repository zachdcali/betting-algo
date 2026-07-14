"""Canonical cooperative lock for the mutable paper-account CSV set."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from functools import wraps
import os
from pathlib import Path
import threading
from typing import Iterator


LOCK_FILENAME = ".operational_csv.lock"
PENDING_RECONCILIATION_TRANSACTION_DIRNAME = ".pending_reconciliation_transaction"
PENDING_RECONCILIATION_TARGET_ROLES = (
    "bets",
    "bankroll",
    "sessions",
    "apply_audit",
)
PENDING_RECONCILIATION_PRIVATE_DIRNAME = ".private"
PENDING_RECONCILIATION_APPLY_AUDIT_FILENAME = (
    "pending_reconciliation_apply_audit.csv"
)
_PROCESS_LOCK = threading.RLock()
_LOCAL = threading.local()


def canonical_operational_lock_path(logs_dir: Path | str) -> Path:
    return Path(logs_dir).resolve() / LOCK_FILENAME


def canonical_pending_reconciliation_transaction_path(
    logs_dir: Path | str,
) -> Path:
    return (
        Path(logs_dir).resolve()
        / PENDING_RECONCILIATION_TRANSACTION_DIRNAME
    )


def canonical_pending_reconciliation_targets(
    logs_dir: Path | str,
) -> tuple[Path, Path, Path, Path]:
    """Return the independently derivable recovery allowlist."""
    logs_dir = Path(logs_dir).resolve()
    return (
        logs_dir / "all_bets.csv",
        logs_dir / "bankroll_history.csv",
        logs_dir / "betting_sessions.csv",
        logs_dir.parent
        / PENDING_RECONCILIATION_PRIVATE_DIRNAME
        / PENDING_RECONCILIATION_APPLY_AUDIT_FILENAME,
    )


def _recover_pending_reconciliation_before_yield(logs_dir: Path) -> None:
    """Repair an interrupted reconciliation before a lock holder sees CSVs.

    Import lazily so normal BetTracker operations do not load the reconciliation
    module.  The import is needed only after an interrupted transaction leaves
    its canonical durable journal behind.
    """
    transaction_dir = canonical_pending_reconciliation_transaction_path(logs_dir)
    if not transaction_dir.exists() and not transaction_dir.is_symlink():
        return

    from operations.pending_reconciliation import (  # local import avoids cycle
        recover_pending_reconciliation_transaction,
    )

    recover_pending_reconciliation_transaction(logs_dir)


@contextmanager
def operational_csv_lock(logs_dir: Path | str) -> Iterator[None]:
    """Take the one process-reentrant, cross-process paper-account lock."""
    lock_path = canonical_operational_lock_path(logs_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        depths = dict(getattr(_LOCAL, "depths", {}))
        key = str(lock_path)
        depth = int(depths.get(key, 0))
        if depth:
            depths[key] = depth + 1
            _LOCAL.depths = depths
            try:
                yield
            finally:
                depths = dict(_LOCAL.depths)
                depths[key] -= 1
                _LOCAL.depths = depths
            return

        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                # Recovery is part of acquiring the canonical lock, not part of
                # the reconciliation command.  Therefore every first-level
                # cooperative reader/writer sees either the complete old file
                # set or a verified complete committed file set, never the
                # partially replaced state left by a dead process.
                _recover_pending_reconciliation_before_yield(lock_path.parent)
                depths[key] = 1
                _LOCAL.depths = depths
                try:
                    yield
                finally:
                    depths = dict(_LOCAL.depths)
                    depths.pop(key, None)
                    _LOCAL.depths = depths
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def locked_operational_csv(method):
    """Serialize a ``BetTracker`` mutation with reconciliation/hydration."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with operational_csv_lock(self.logs_dir):
            return method(self, *args, **kwargs)

    return wrapped
