"""Resolve exact feature snapshots across immutable and derived CSV stores.

``logs/features_*.csv`` is the immutable per-run evidence.  The aggregate
``logs/feature_vectors.csv`` and Supabase ``dash_features`` projection are
derived copies.  A shared resolver keeps every consumer from independently
choosing or hashing those copies:

* immutable rows are parsed with pandas' round-trip float parser and win for an
  existing ``feature_snapshot_id``;
* duplicate identities must agree on run, match, and ordered schema;
* a derived copy may differ only within the explicit 1e-12 element-wise
  tolerance, while conflicting immutable rows must retain the exact same v1
  vector hash;
* material disagreement raises instead of silently changing an evaluation,
  import, or dashboard cohort.

The tolerance is solely a duplicate-reconciliation rule.  It does not change
``feature_contract.vector_sha256`` or the ``base_141@1.0.0`` representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:
    from feature_contract import normalize_feature_vector, vector_sha256
except ImportError:  # pragma: no cover - package-style execution
    from production.feature_contract import (  # type: ignore
        normalize_feature_vector,
        vector_sha256,
    )


FEATURE_DUPLICATE_TOLERANCE = 1e-12


class FeatureLineageConflict(RuntimeError):
    """Raised when one snapshot ID maps to contradictory exact evidence."""


def expected_feature_hash_matches(
    expected_hash: Any,
    accepted_hashes: Iterable[str],
) -> bool:
    """Apply the shared referential rule for an operational hash claim."""
    expected = _text(expected_hash)
    return not expected or expected in set(accepted_hashes)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _integer(value: Any) -> int | None:
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def read_feature_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read feature evidence without pandas' lossy fast-float conversion."""
    options = {
        "keep_default_na": False,
        "low_memory": False,
        "float_precision": "round_trip",
    }
    options.update(kwargs)
    return pd.read_csv(path, **options)


def vectors_allclose(
    left: Mapping[str, float],
    right: Mapping[str, float],
    ordered_names: Sequence[str],
) -> bool:
    """Return whether every ordered field agrees within the lineage tolerance."""
    return all(
        name in left
        and name in right
        and math.isclose(
            float(left[name]),
            float(right[name]),
            rel_tol=FEATURE_DUPLICATE_TOLERANCE,
            abs_tol=FEATURE_DUPLICATE_TOLERANCE,
        )
        for name in ordered_names
    )


@dataclass(frozen=True)
class FeatureOccurrence:
    snapshot_id: str
    run_id: str
    match_uid: str
    schema_sha256: str
    stored_schema_sha256: str
    stored_vector_sha256: str
    raw_vector_sha256: str
    verified_vector_sha256: str
    feature_count: int
    vector: Mapping[str, float]
    payload_names: frozenset[str]
    payload_numeric: Mapping[str, float]
    payload_invalid: Mapping[str, str]
    structural_issues: tuple[str, ...]
    metadata_issues: tuple[str, ...]
    status: str
    source_kind: str
    source_file: str
    source_row: int
    row: Mapping[str, Any]

    @property
    def location(self) -> tuple[str, int]:
        return self.source_file, self.source_row

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.run_id, self.match_uid, self.schema_sha256

    @property
    def structurally_verified(self) -> bool:
        return bool(
            self.status == "ok"
            and self.verified_vector_sha256
            and not self.structural_issues
            and not self.metadata_issues
        )


@dataclass(frozen=True)
class FeatureLineageResolution:
    canonical_by_id: Mapping[str, FeatureOccurrence]
    occurrences_by_id: Mapping[str, tuple[FeatureOccurrence, ...]]
    immutable_ids: frozenset[str]
    invalid_ids: frozenset[str]

    def canonical_location(self, snapshot_id: str) -> tuple[str, int] | None:
        occurrence = self.canonical_by_id.get(snapshot_id)
        return occurrence.location if occurrence is not None else None

    def referential_vector_hashes(self, snapshot_id: str) -> frozenset[str]:
        """Return the sole exact hash allowed for prediction referential checks.

        The tolerance reconciles duplicate storage copies only. A different
        derived-copy hash never becomes decision-grade operational lineage.
        """
        authority = self.canonical_by_id.get(snapshot_id)
        if authority is None or not authority.structurally_verified:
            return frozenset()
        return frozenset({authority.verified_vector_sha256})


def _payload(
    row: Mapping[str, Any],
    ordered_names: Sequence[str],
    *,
    source_kind: str,
) -> Mapping[str, Any]:
    if source_kind == "derived":
        raw = _text(row.get("features_json"))
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {name: row.get(name) for name in ordered_names}


def parse_feature_occurrence(
    row: Mapping[str, Any],
    ordered_names: Sequence[str],
    *,
    source_kind: str,
    source_file: str,
    source_row: int,
) -> FeatureOccurrence | None:
    """Parse one exact-ID row while preserving its source metadata."""
    if source_kind not in {"immutable", "derived"}:
        raise ValueError(f"unknown feature source kind: {source_kind}")
    snapshot_id = _text(row.get("feature_snapshot_id"))
    if not snapshot_id:
        return None

    names = tuple(ordered_names)
    schema_sha256 = sha256("\x1f".join(names).encode("utf-8")).hexdigest()
    raw_payload = _payload(row, names, source_kind=source_kind)
    vector, structural_issues = normalize_feature_vector(raw_payload, names)
    payload_names: set[str] = set()
    payload_numeric: dict[str, float] = {}
    payload_invalid: dict[str, str] = {}
    for name in names:
        if name not in raw_payload:
            continue
        payload_names.add(name)
        value = raw_payload[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            payload_invalid[name] = json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            )
            continue
        if math.isfinite(numeric):
            payload_numeric[name] = numeric
        else:
            payload_invalid[name] = str(value)
    has_full_finite_vector = len(vector) == len(names)
    raw_hash = vector_sha256(vector, names) if has_full_finite_vector else ""
    stored_schema = _text(row.get("feature_schema_sha256"))
    stored_hash = _text(row.get("feature_vector_sha256"))
    stored_count_text = _text(row.get("feature_count"))
    stored_count = _integer(row.get("feature_count"))
    metadata_issues: list[str] = []
    if stored_schema and stored_schema != schema_sha256:
        metadata_issues.append("source_schema_hash_mismatch")
    if stored_count_text and stored_count is None:
        metadata_issues.append("source_feature_count_invalid")
    elif stored_count is not None and stored_count != len(names):
        metadata_issues.append("source_feature_count_mismatch")
    if stored_hash and (not raw_hash or stored_hash != raw_hash):
        metadata_issues.append("source_vector_hash_mismatch")

    status = _text(row.get("build_status") or row.get("status") or "unknown").lower()
    verified_hash = ""
    if (
        status == "ok"
        and not structural_issues
        and not metadata_issues
        and raw_hash
    ):
        verified_hash = stored_hash or raw_hash
    return FeatureOccurrence(
        snapshot_id=snapshot_id,
        run_id=_text(row.get("run_id")),
        match_uid=_text(row.get("match_uid")),
        schema_sha256=schema_sha256,
        stored_schema_sha256=stored_schema,
        stored_vector_sha256=stored_hash,
        raw_vector_sha256=raw_hash,
        verified_vector_sha256=verified_hash,
        feature_count=len(vector),
        vector=vector,
        payload_names=frozenset(payload_names),
        payload_numeric=payload_numeric,
        payload_invalid=payload_invalid,
        structural_issues=tuple(structural_issues),
        metadata_issues=tuple(metadata_issues),
        status=status,
        source_kind=source_kind,
        source_file=source_file,
        source_row=int(source_row),
        row=dict(row),
    )


def _conflict(snapshot_id: str, reason: str, *rows: FeatureOccurrence) -> None:
    locations = ", ".join(
        f"{row.source_file}:{row.source_row}" for row in rows
    )
    raise FeatureLineageConflict(
        f"feature_snapshot_id {snapshot_id!r} {reason}: {locations}"
    )


def _require_identity(
    authority: FeatureOccurrence,
    candidate: FeatureOccurrence,
) -> None:
    if authority.identity != candidate.identity:
        _conflict(
            authority.snapshot_id,
            "has conflicting run/match/schema identity",
            authority,
            candidate,
        )


def _raw_payloads_allclose(
    authority: FeatureOccurrence,
    candidate: FeatureOccurrence,
) -> bool:
    """Compare even partial/invalid status=ok payloads without losing fields."""
    if authority.payload_names != candidate.payload_names:
        return False
    if authority.payload_invalid != candidate.payload_invalid:
        return False
    if set(authority.payload_numeric) != set(candidate.payload_numeric):
        return False
    return all(
        math.isclose(
            authority.payload_numeric[name],
            candidate.payload_numeric[name],
            rel_tol=FEATURE_DUPLICATE_TOLERANCE,
            abs_tol=FEATURE_DUPLICATE_TOLERANCE,
        )
        for name in authority.payload_numeric
    )


def _raw_payload_overlap_compatible(
    authority: FeatureOccurrence,
    candidate: FeatureOccurrence,
) -> bool:
    """Validate fields retained by both sparse non-decision-grade copies."""
    for name in authority.payload_names & candidate.payload_names:
        authority_numeric = authority.payload_numeric.get(name)
        candidate_numeric = candidate.payload_numeric.get(name)
        if authority_numeric is not None and candidate_numeric is not None:
            if not math.isclose(
                authority_numeric,
                candidate_numeric,
                rel_tol=FEATURE_DUPLICATE_TOLERANCE,
                abs_tol=FEATURE_DUPLICATE_TOLERANCE,
            ):
                return False
            continue
        if authority.payload_invalid.get(name) != candidate.payload_invalid.get(name):
            return False
    return True


def _require_immutable_duplicate(
    authority: FeatureOccurrence,
    candidate: FeatureOccurrence,
    ordered_names: Sequence[str],
) -> None:
    _require_identity(authority, candidate)
    if authority.status != candidate.status:
        _conflict(
            authority.snapshot_id,
            "has conflicting immutable build status",
            authority,
            candidate,
        )
    if authority.metadata_issues or candidate.metadata_issues:
        _conflict(
            authority.snapshot_id,
            "has invalid immutable hash/schema metadata",
            authority,
            candidate,
        )
    if authority.status != "ok":
        # Historical pipeline runs incorrectly assigned exact-looking IDs to
        # skip/guard diagnostics.  The same match could therefore emit two
        # different placeholder payloads under one ID (for example structural
        # round failure followed by the pre-start time guard).  They are never
        # decision-grade and remain in ``invalid_ids``; permit hydration only
        # when their identity and non-ok status agree and neither row claims an
        # exact stored vector hash.
        if authority.stored_vector_sha256 or candidate.stored_vector_sha256:
            _conflict(
                authority.snapshot_id,
                "has an exact hash claim on a non-decision-grade immutable row",
                authority,
                candidate,
            )
        return
    if not vectors_allclose(authority.vector, candidate.vector, ordered_names):
        _conflict(
            authority.snapshot_id,
            "has materially divergent immutable vectors",
            authority,
            candidate,
        )
    # Immutable duplicates are not alternative rounding authorities.  Their
    # serialized base_141 payload must retain the same exact v1 hash.
    if authority.raw_vector_sha256 != candidate.raw_vector_sha256:
        _conflict(
            authority.snapshot_id,
            "has conflicting immutable v1 vector hashes",
            authority,
            candidate,
        )


def _require_derived_copy(
    authority: FeatureOccurrence,
    candidate: FeatureOccurrence,
    ordered_names: Sequence[str],
) -> None:
    _require_identity(authority, candidate)
    if authority.status != candidate.status:
        _conflict(
            authority.snapshot_id,
            "has conflicting derived build status",
            authority,
            candidate,
        )
    hard_metadata_issues = set(candidate.metadata_issues) - {
        "source_vector_hash_mismatch"
    }
    if hard_metadata_issues:
        _conflict(
            authority.snapshot_id,
            "has conflicting derived schema/count metadata",
            authority,
            candidate,
        )

    authority_has_vector = len(authority.vector) == len(ordered_names)
    candidate_has_vector = len(candidate.vector) == len(ordered_names)
    if authority_has_vector and candidate_has_vector and not vectors_allclose(
        authority.vector, candidate.vector, ordered_names
    ):
        _conflict(
            authority.snapshot_id,
            "has materially divergent derived vector",
            authority,
            candidate,
        )
    if authority.status == "ok" and not _raw_payloads_allclose(
        authority, candidate
    ):
        _conflict(
            authority.snapshot_id,
            "has contradictory invalid/partial derived payloads",
            authority,
            candidate,
        )

    if authority.status != "ok":
        # Skip/error rows are operational evidence, not exact model vectors.
        # The historical aggregate often omitted the ordered zeros from these
        # rows. That sparse copy is acceptable only when neither side claims a
        # decision-grade vector hash. Full vectors, when both are present, were
        # still compared above so a duplicate ID cannot hide divergence.
        if authority.stored_vector_sha256 or candidate.stored_vector_sha256:
            _conflict(
                authority.snapshot_id,
                "has an exact hash claim on a non-decision-grade copy",
                authority,
                candidate,
            )
        if not _raw_payload_overlap_compatible(authority, candidate):
            _conflict(
                authority.snapshot_id,
                "has contradictory sparse non-decision-grade payloads",
                authority,
                candidate,
            )
        return

    if not authority.structurally_verified:
        if authority_has_vector != candidate_has_vector:
            _conflict(
                authority.snapshot_id,
                "has missing derived vector fields for invalid authority",
                authority,
                candidate,
            )
        candidate_claims_exactness = bool(
            candidate.stored_vector_sha256
            or (
                candidate_has_vector
                and not candidate.structural_issues
                and not hard_metadata_issues
            )
        )
        if candidate_claims_exactness:
            _conflict(
                authority.snapshot_id,
                "derived copy cannot rehabilitate invalid immutable authority",
                authority,
                candidate,
            )
        return

    if authority_has_vector != candidate_has_vector:
        _conflict(
            authority.snapshot_id,
            "has missing derived vector fields",
            authority,
            candidate,
        )
    if not candidate_has_vector:
        _conflict(
            authority.snapshot_id,
            "has missing derived vector fields",
            authority,
            candidate,
        )
    if candidate.stored_vector_sha256 and candidate.stored_vector_sha256 not in {
        candidate.raw_vector_sha256,
        authority.verified_vector_sha256,
    }:
        _conflict(
            authority.snapshot_id,
            "has unrelated derived vector hash",
            authority,
            candidate,
        )


def resolve_feature_lineage(
    *,
    ordered_names: Sequence[str],
    immutable_sources: Iterable[tuple[str, pd.DataFrame]] = (),
    derived_sources: Iterable[tuple[str, pd.DataFrame]] = (),
) -> FeatureLineageResolution:
    """Resolve canonical evidence for every immutable snapshot ID."""
    names = tuple(ordered_names)
    occurrences: dict[str, list[FeatureOccurrence]] = {}

    def collect(source_kind: str, sources: Iterable[tuple[str, pd.DataFrame]]) -> None:
        for source_file, frame in sources:
            if frame.empty or "feature_snapshot_id" not in frame.columns:
                continue
            for position, (_, row) in enumerate(frame.iterrows(), start=2):
                occurrence = parse_feature_occurrence(
                    row,
                    names,
                    source_kind=source_kind,
                    source_file=source_file,
                    source_row=position,
                )
                if occurrence is not None:
                    occurrences.setdefault(occurrence.snapshot_id, []).append(occurrence)

    collect("immutable", immutable_sources)
    collect("derived", derived_sources)

    canonical: dict[str, FeatureOccurrence] = {}
    immutable_ids: set[str] = set()
    invalid_ids: set[str] = set()
    frozen_occurrences: dict[str, tuple[FeatureOccurrence, ...]] = {}
    for snapshot_id in sorted(occurrences):
        rows = sorted(
            occurrences[snapshot_id],
            key=lambda item: (
                0 if item.source_kind == "immutable" else 1,
                item.source_file,
                item.source_row,
            ),
        )
        immutable = [row for row in rows if row.source_kind == "immutable"]
        derived = [row for row in rows if row.source_kind == "derived"]
        authority = immutable[0] if immutable else derived[0]
        if immutable:
            immutable_ids.add(snapshot_id)
            for candidate in immutable[1:]:
                _require_immutable_duplicate(authority, candidate, names)
            for candidate in derived:
                _require_derived_copy(authority, candidate, names)
        else:
            for candidate in derived[1:]:
                _require_identity(authority, candidate)
                if (
                    authority.status != candidate.status
                    or authority.raw_vector_sha256 != candidate.raw_vector_sha256
                    or (
                        authority.status == "ok"
                        and not _raw_payloads_allclose(authority, candidate)
                    )
                    or (
                        authority.status != "ok"
                        and not _raw_payload_overlap_compatible(authority, candidate)
                    )
                    or authority.metadata_issues
                    or candidate.metadata_issues
                ):
                    _conflict(
                        snapshot_id,
                        "has contradictory derived copies without immutable authority",
                        authority,
                        candidate,
                    )
        canonical[snapshot_id] = authority
        frozen_occurrences[snapshot_id] = tuple(rows)
        if not authority.structurally_verified:
            invalid_ids.add(snapshot_id)

    return FeatureLineageResolution(
        canonical_by_id=canonical,
        occurrences_by_id=frozen_occurrences,
        immutable_ids=frozenset(immutable_ids),
        invalid_ids=frozenset(invalid_ids),
    )


def validate_derived_feature_copy(
    authority_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    ordered_names: Sequence[str],
    *,
    snapshot_id: str,
    authority_source: str = "authoritative projection",
    candidate_source: str = "derived projection",
    require_valid_authority: bool = True,
) -> None:
    """Validate a dashboard/aggregate copy against a resolved authority row."""
    authority = parse_feature_occurrence(
        authority_row,
        ordered_names,
        source_kind="derived",
        source_file=authority_source,
        source_row=2,
    )
    candidate = parse_feature_occurrence(
        candidate_row,
        ordered_names,
        source_kind="derived",
        source_file=candidate_source,
        source_row=2,
    )
    if authority is None or candidate is None:
        raise FeatureLineageConflict(
            f"feature_snapshot_id {snapshot_id!r} is missing from a derived copy"
        )
    if authority.snapshot_id != snapshot_id or candidate.snapshot_id != snapshot_id:
        raise FeatureLineageConflict(
            f"feature_snapshot_id {snapshot_id!r} changed in a derived copy"
        )
    if require_valid_authority:
        validate_derived_projection_authority(authority)
    _require_derived_copy(authority, candidate, tuple(ordered_names))


def validate_derived_projection_authority(authority: FeatureOccurrence) -> None:
    """Reject an invalid derived-only row that nevertheless claims exactness."""
    explicitly_complete = _text(authority.row.get("features_complete")).lower() in {
        "true", "1", "1.0", "t", "yes",
    }
    claims_exactness = bool(
        explicitly_complete
        or authority.stored_vector_sha256
        or authority.metadata_issues
    )
    if authority.status != "ok":
        if claims_exactness:
            _conflict(
                authority.snapshot_id,
                "has an exactness claim on a non-decision-grade projection authority",
                authority,
            )
        return
    if authority.structurally_verified:
        return
    if claims_exactness:
        _conflict(
            authority.snapshot_id,
            "has an invalid decision-grade projection authority",
            authority,
        )


def load_production_feature_lineage(
    production_dir: str | Path,
    ordered_names: Sequence[str],
    *,
    include_run_feature_files: bool = True,
) -> FeatureLineageResolution:
    """Load the on-disk aggregate and immutable run files through one contract."""
    production = Path(production_dir).resolve()
    immutable_sources: list[tuple[str, pd.DataFrame]] = []
    if include_run_feature_files:
        for path in sorted((production / "logs").glob("features_*.csv")):
            relative = path.relative_to(production).as_posix()
            immutable_sources.append((relative, read_feature_csv(path)))
    derived_sources: list[tuple[str, pd.DataFrame]] = []
    aggregate = production / "logs" / "feature_vectors.csv"
    if aggregate.exists():
        derived_sources.append((
            aggregate.relative_to(production).as_posix(),
            read_feature_csv(aggregate),
        ))
    return resolve_feature_lineage(
        ordered_names=ordered_names,
        immutable_sources=immutable_sources,
        derived_sources=derived_sources,
    )
