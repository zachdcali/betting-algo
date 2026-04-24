#!/usr/bin/env python3
"""Validate that model_registry.json points at real artifacts."""

from __future__ import annotations

import sys

try:
    from registry_utils import validate_registry
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .registry_utils import validate_registry


def main() -> int:
    issues = validate_registry()
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
