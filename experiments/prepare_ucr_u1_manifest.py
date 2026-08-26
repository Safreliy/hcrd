"""Freeze the label-independent population and split for UCR protocol U1."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import aeon
import numpy as np
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length


PROJECT = Path(__file__).resolve().parents[1]


def assignment(name: str) -> str:
    """Hash-based assignment fixed in the protocol."""

    return (
        "discovery"
        if hashlib.sha256(name.encode("utf-8")).digest()[0] < 128
        else "confirmation"
    )


def exclusion_reason(
    *, train_size: int, test_size: int, length: int, train_classes: int, test_classes: int
) -> str | None:
    total_cases = train_size + test_size
    if length < 16:
        return "length_below_16"
    if length > 2048:
        return "length_above_2048"
    if total_cases > 10_000:
        return "case_count_above_10000"
    if total_cases * length > 10_000_000:
        return "scalar_count_above_10000000"
    if train_size < 20:
        return "training_size_below_20"
    if train_classes < 2 or test_classes < 2:
        return "split_has_fewer_than_two_classes"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=PROJECT / "data" / "raw" / "ucr_2018"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "data" / "manifests" / "ucr_u1_manifest.json",
    )
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index, name in enumerate(sorted(univariate_equal_length), start=1):
        train_x, train_y = load_classification(
            name=name, split="train", extract_path=args.data
        )
        test_x, test_y = load_classification(
            name=name, split="test", extract_path=args.data
        )
        if not (
            isinstance(train_x, np.ndarray)
            and isinstance(test_x, np.ndarray)
            and train_x.ndim == 3
            and test_x.ndim == 3
            and train_x.shape[1] == test_x.shape[1] == 1
            and train_x.shape[2] == test_x.shape[2]
        ):
            raise RuntimeError(f"unexpected collection layout for {name}")
        length = int(train_x.shape[2])
        reason = exclusion_reason(
            train_size=int(train_x.shape[0]),
            test_size=int(test_x.shape[0]),
            length=length,
            train_classes=int(np.unique(train_y).size),
            test_classes=int(np.unique(test_y).size),
        )
        record = {
            "name": name,
            "name_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            "assignment": assignment(name),
            "train_size": int(train_x.shape[0]),
            "test_size": int(test_x.shape[0]),
            "length": length,
            "class_count_train": int(np.unique(train_y).size),
            "class_count_test": int(np.unique(test_y).size),
            "scalar_count": int((train_x.shape[0] + test_x.shape[0]) * length),
            "eligible": reason is None,
            "exclusion_reason": reason,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(univariate_equal_length),
                    **record,
                }
            ),
            flush=True,
        )

    manifest = {
        "protocol": "U1 / frozen before classifier outcomes",
        "archive": "UCR Time Series Classification Archive 2018",
        "archive_population": "aeon 1.5.0 univariate_equal_length (112 problems)",
        "source": "https://www.timeseriesclassification.com/",
        "assignment_rule": (
            "discovery iff first byte of SHA256(UTF8(dataset_name)) is below 128"
        ),
        "resource_rules": {
            "minimum_length": 16,
            "maximum_length": 2048,
            "maximum_cases": 10_000,
            "maximum_scalar_observations": 10_000_000,
            "minimum_training_cases": 20,
            "minimum_classes_each_split": 2,
        },
        "python": platform.python_version(),
        "aeon": aeon.__version__,
        "numpy": np.__version__,
        "records": records,
        "counts": {
            "population": len(records),
            "eligible": sum(bool(record["eligible"]) for record in records),
            "discovery_eligible": sum(
                bool(record["eligible"]) and record["assignment"] == "discovery"
                for record in records
            ),
            "confirmation_eligible": sum(
                bool(record["eligible"]) and record["assignment"] == "confirmation"
                for record in records
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"manifest": str(args.output), "sha256": digest, **manifest["counts"]}))


if __name__ == "__main__":
    main()

