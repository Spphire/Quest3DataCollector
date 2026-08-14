#!/usr/bin/env python3
"""Build a deterministic Collector recording manifest from a user-specified batch."""

from __future__ import annotations

import argparse
from pathlib import Path


def recording_root(input_dir: Path) -> Path:
    direct = sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("record_"))
    if direct:
        return input_dir
    nested = input_dir / "pc_recordings"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"No record_* directories under {input_dir} or {nested}")


def load_requested_ids(path: Path) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"Requested record list is empty: {path}")
    return values


def build_manifest(
    input_dir: Path,
    requested_ids: set[str] | None,
    start_record_id: str | None,
    end_record_id: str | None,
) -> tuple[Path, list[str]]:
    root = recording_root(input_dir.resolve())
    available = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("record_")
    }
    if requested_ids is not None:
        missing = sorted(requested_ids - available)
        if missing:
            raise FileNotFoundError(f"Requested recording directories are missing: {missing}")
        available &= requested_ids
    selected = sorted(
        record_id
        for record_id in available
        if (start_record_id is None or record_id >= start_record_id)
        and (end_record_id is None or record_id <= end_record_id)
    )
    if not selected:
        raise ValueError("The batch selectors matched no recording directories")
    return root, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requested-record-ids-file",
        type=Path,
        help="Optional explicit input list; comments and blank lines are ignored.",
    )
    parser.add_argument("--start-record-id", help="Inclusive lexicographic record ID boundary.")
    parser.add_argument("--end-record-id", help="Inclusive lexicographic record ID boundary.")
    args = parser.parse_args()

    requested_ids = (
        load_requested_ids(args.requested_record_ids_file.resolve())
        if args.requested_record_ids_file
        else None
    )
    root, record_ids = build_manifest(
        input_dir=args.input_dir,
        requested_ids=requested_ids,
        start_record_id=args.start_record_id,
        end_record_id=args.end_record_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# Collector recording batch manifest.",
                f"# source={root}",
                f"# count={len(record_ids)}",
                *record_ids,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"source={root} selected={len(record_ids)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
