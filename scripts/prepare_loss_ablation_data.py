#!/usr/bin/env python3
"""Generate fresh test layouts and verify unchanged pilot train/val arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ant_homing.level1.data import generate_dataset, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, metadata = load_dataset(args.pilot)
    scenes = dict(metadata["scene_seeds"])
    scenes["test"] = list(range(400, 412))
    generate_dataset(args.output, seed=metadata["seed"], scene_seeds=scenes)
    arrays, fresh = load_dataset(args.output)
    for key in original:
        split_key = "query_splits" if key.startswith("query") else "gallery_splits"
        before = original[key][original[split_key] != "test"]
        after = arrays[key][arrays[split_key] != "test"]
        if before.dtype != after.dtype or not np.array_equal(before, after):
            raise ValueError(f"Pilot train/validation arrays differ: {key}")
    record = {
        "training_validation_arrays": "byte-identical logical arrays to original pilot",
        "compared_keys": list(original),
        "original_dataset_sha256": metadata["data_sha256"],
        "dataset_sha256": fresh["data_sha256"],
        "test_scene_seeds": scenes["test"],
    }
    (args.output / "identity-check.json").write_text(json.dumps(record, indent=2) + "\n")
    print("All training/validation arrays and dtypes match the pilot.")


if __name__ == "__main__":
    main()
