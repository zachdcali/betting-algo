#!/usr/bin/env python3
"""Close the latest non-terminal run after an external process interruption."""

from __future__ import annotations

import argparse

from audit_logger import finalize_latest_running_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="cancelled")
    parser.add_argument(
        "--reason",
        default="GitHub Actions pipeline step ended before terminal publication",
    )
    args = parser.parse_args()
    run_id = finalize_latest_running_run(
        status=args.status,
        error_message=args.reason,
    )
    if run_id:
        print(f"Finalized interrupted pipeline attempt: {run_id} ({args.status})")
    else:
        print("No running pipeline attempt required finalization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
