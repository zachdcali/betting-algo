#!/usr/bin/env python3
"""Validate that model_registry.json points at real artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Calibrated sklearn pickles reference ``models.nn_runtime``.  Direct script
# execution otherwise exposes only production/models on sys.path, so add the
# package root before registry validation attempts to deserialize one.
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
if str(PRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_ROOT))

try:
    from models.registry_utils import validate_registry
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .registry_utils import validate_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate model-registry contracts and artifact files.",
    )
    parser.add_argument(
        "--artifact-scope",
        choices=("promoted", "all"),
        default="promoted",
        help=(
            "Require promoted inference artifacts (default), or every "
            "registered archive and candidate artifact for a local audit."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = validate_registry(artifact_scope=args.artifact_scope)
    missing = issues.get("missing", [])
    invalid = issues.get("invalid", [])

    if not missing and not invalid:
        print("✅ model_registry.json is consistent")
        return 0

    if invalid:
        print("Invalid entries:")
        for issue in invalid:
            print(f"  - {issue}")

    if missing:
        print("Missing artifacts:")
        for issue in missing:
            print(f"  - {issue}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
