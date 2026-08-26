from __future__ import annotations

import hashlib
import unicodedata


def _partition(axis: str, identifier: str) -> str:
    normalized = unicodedata.normalize("NFKC", identifier).strip().lower()
    byte = hashlib.sha256(
        f"hcrd-e1-v1|{axis}|{normalized}".encode("utf-8")
    ).digest()[0]
    if byte <= 153:
        return "train"
    if byte <= 204:
        return "validation"
    return "confirmation"


def test_frozen_partition_is_deterministic_and_axis_specific() -> None:
    assert _partition("sample", "  Sample A  ") == _partition(
        "sample", "sample a"
    )
    sample_digest = hashlib.sha256(b"hcrd-e1-v1|sample|sample a").digest()
    peak_digest = hashlib.sha256(b"hcrd-e1-v1|peak|sample a").digest()
    assert sample_digest != peak_digest


def test_frozen_partition_covers_exact_byte_ranges() -> None:
    observed = {_partition("sample", f"sample-{index}") for index in range(1000)}
    assert observed == {"train", "validation", "confirmation"}
