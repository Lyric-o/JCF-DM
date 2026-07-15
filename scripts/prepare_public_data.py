#!/usr/bin/env python3
"""Download the public SUPPORT and Rotterdam--GBSG benchmark tables."""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = {
    "support": (8873, 16),
    "gbsg": (2232, 9),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from pycox import datasets
    except ImportError as exc:
        raise SystemExit(
            "pycox is required for data acquisition; run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    args.data_dir.mkdir(parents=True, exist_ok=True)
    sources = {"support": datasets.support, "gbsg": datasets.gbsg}
    for name, source in sources.items():
        destination = args.data_dir / f"{name}.csv"
        if destination.exists() and not args.overwrite:
            print(f"kept existing {destination}")
            continue
        frame = source.read_df().copy()
        expected_shape = EXPECTED[name]
        if frame.shape != expected_shape:
            raise RuntimeError(
                f"{name}: expected shape {expected_shape}, got {frame.shape}; "
                "check the pycox dataset version before continuing"
            )
        required = {"duration", "event"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{name}: missing required columns {sorted(required)}")
        frame.to_csv(destination, index=False)
        print(f"wrote {destination} with shape {frame.shape}")


if __name__ == "__main__":
    main()
