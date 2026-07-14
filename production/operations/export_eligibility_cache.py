"""Export one accepted, generation-and-seal-pinned eligibility profile bundle.

This command is a read-only operational database client.  It refuses to
publish when the accepted projection is stale, ambiguous, expired, or does not
match the exact configured generation and content seal.  The resulting local
bundle is a short-lived derived read model, never an authority.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from typing import Any, Iterable

from production.eligibility_cache import (
    DEFAULT_MAX_AGE,
    build_profile_bundle,
    write_profile_bundle_export,
)
from production.storage.eligibility import (
    ELIGIBILITY_GENERATION_ENV,
    ELIGIBILITY_PROJECTION_SEAL_ENV,
    EligibilityContractError,
    EligibilityProjectionReader,
)


NAME_CONFLICT_KINDS = (
    "player_identity",
    "player_alias",
    "player_source_identity",
    "player_source_key",
    "player_name_binding",
)


def _read_projection(
    connection: Any,
    *,
    generation_sha256: str,
    projection_seal_sha256: str,
) -> tuple[list[dict[str, Any]], int]:
    reader = EligibilityProjectionReader(
        connection,
        expected_generation_sha256=generation_sha256,
        expected_projection_seal_sha256=projection_seal_sha256,
    )
    reader.verify_contract()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT evidence_kind, subject_key
                 FROM api.eligibility_conflicts
                WHERE eligibility_generation_sha256 = %s
                  AND evidence_kind = ANY(%s)
                ORDER BY evidence_kind, subject_key
                LIMIT 25""",
            (generation_sha256, list(NAME_CONFLICT_KINDS)),
        )
        conflicts = list(cursor.fetchall())
        if conflicts:
            sample = ", ".join(f"{kind}:{key}" for kind, key in conflicts[:5])
            raise EligibilityContractError(
                "accepted eligibility names are ambiguous; refusing whole-bundle "
                f"export ({sample})"
            )
        cursor.execute(
            """SELECT b.player_name_norm,
                      b.canonical_player_id,
                      p.field_name,
                      p.field_value,
                      b.valid_until AS binding_valid_until,
                      p.valid_until AS profile_valid_until
                 FROM api.current_player_name_bindings b
                 JOIN api.current_player_profiles p
                   ON p.eligibility_generation_sha256 = b.eligibility_generation_sha256
                  AND p.canonical_player_id = b.canonical_player_id
                WHERE b.eligibility_generation_sha256 = %s
                  AND p.field_name IN ('height_cm', 'hand')
                ORDER BY b.player_name_norm, p.field_name""",
            (generation_sha256,),
        )
        rows = [
            {
                "player_name_norm": row[0],
                "canonical_player_id": row[1],
                "field_name": row[2],
                "field_value": row[3],
                "binding_valid_until": row[4],
                "profile_valid_until": row[5],
            }
            for row in cursor.fetchall()
        ]
    return rows, reader.projection_row_count


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the accepted ops eligibility profile bundle",
    )
    parser.add_argument(
        "--database-url-env",
        default="OPERATIONAL_STAGING_DATABASE_URL",
        help="name of the explicitly configured staging/cutover database URL",
    )
    parser.add_argument(
        "--generation-sha256",
        default=os.environ.get(ELIGIBILITY_GENERATION_ENV, ""),
    )
    parser.add_argument(
        "--projection-seal-sha256",
        default=os.environ.get(ELIGIBILITY_PROJECTION_SEAL_ENV, ""),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=int(DEFAULT_MAX_AGE.total_seconds()),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    generation = str(args.generation_sha256 or "").strip().lower()
    seal = str(args.projection_seal_sha256 or "").strip().lower()
    if not generation:
        parser.error("--generation-sha256 is required")
    if not seal:
        parser.error("--projection-seal-sha256 is required")
    if args.max_age_seconds <= 0:
        parser.error("--max-age-seconds must be positive")
    database_url = os.environ.get(args.database_url_env, "").strip()
    if not database_url:
        parser.error(f"{args.database_url_env} is not configured")

    import psycopg

    with psycopg.connect(database_url) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        rows, projection_row_count = _read_projection(
            connection,
            generation_sha256=generation,
            projection_seal_sha256=seal,
        )
    data = build_profile_bundle(rows)
    manifest = write_profile_bundle_export(
        output_dir=args.output_dir,
        generation_sha256=generation,
        projection_seal_sha256=seal,
        projection_row_count=projection_row_count,
        data=data,
        max_age=timedelta(seconds=args.max_age_seconds),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
