#!/usr/bin/env python3
"""Validate dimensions, schemas, and reference fingerprints of local data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


SPECS = {
    "support.csv": {
        "shape": (8873, 16),
        "required": {"duration", "event"},
        "sha256": "17a7aca1b760004e991998bdf472eecadbfcad31902de1e4ee022b95b7d2b2d6",
        "restricted": False,
    },
    "gbsg.csv": {
        "shape": (2232, 9),
        "required": {"duration", "event"},
        "sha256": "d9b457518036da5acab0741f9bd6872f12e5a7a6b9c7c00860ffc8f779965d0b",
        "restricted": False,
    },
    "seer.csv": {
        "shape": (11600, 63),
        "required": {"time", "cod"},
        "sha256": "c57e9841391c941549bdc43dc906ce63de25800c979a8096f87aa9a0000a6373",
        "restricted": True,
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--require-seer", action="store_true")
    args = parser.parse_args()

    failures = []
    for filename, spec in SPECS.items():
        path = args.data_dir / filename
        if not path.exists():
            if filename == "seer.csv" and not args.require_seer:
                print("SKIP seer.csv (restricted; public-only reproduction remains available)")
                continue
            failures.append(f"missing {path}")
            continue
        frame = pd.read_csv(path)
        if frame.shape != spec["shape"]:
            failures.append(f"{filename}: expected {spec['shape']}, got {frame.shape}")
        missing = spec["required"] - set(frame.columns)
        if missing:
            failures.append(f"{filename}: missing columns {sorted(missing)}")
        observed_hash = digest(path)
        if observed_hash == spec["sha256"]:
            print(f"OK   {filename}: shape={frame.shape}, reference hash matched")
        elif spec["restricted"]:
            failures.append(f"{filename}: exact restricted-cohort hash did not match")
        else:
            print(f"WARN {filename}: schema matched but byte hash differed")

    if failures:
        raise SystemExit("data validation failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
