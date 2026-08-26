from __future__ import annotations

from pathlib import Path

import numpy as np

from hcrd.lcms_data import EICFlatCache


def test_flat_cache_respects_sample_major_order(tmp_path: Path) -> None:
    (tmp_path / "cache_manifest.txt").write_text(
        "\n".join(
            [
                "format=hcrd-e1-flat-v1",
                "sample_count=2",
                "peak_count=2",
                "case_count=4",
                "short_max_length=3",
                "long_max_length=4",
                "dtype=float32-little-endian",
                "ordering=sample-major-then-peak-major",
                "rows=intensity,retention_time",
            ]
        )
        + "\n"
    )
    (tmp_path / "sample_names.txt").write_text("s0\ns1\n")
    (tmp_path / "peak_names.txt").write_text("p0\np1\n")
    short = np.arange(12, dtype="<f4").reshape(4, 3)
    long = np.arange(16, dtype="<f4").reshape(4, 4) + 100
    short.tofile(tmp_path / "short_intensity.f32")
    (short + 20).tofile(tmp_path / "short_retention_time.f32")
    long.tofile(tmp_path / "long_intensity.f32")
    (long + 20).tofile(tmp_path / "long_retention_time.f32")
    np.asarray([2, 3, 1, 2], dtype="<i4").tofile(tmp_path / "short_length.i32")
    np.asarray([3, 4, 2, 1], dtype="<i4").tofile(tmp_path / "long_length.i32")
    cache = EICFlatCache(tmp_path)
    pair = cache.pair(1, 0)
    assert pair.short_intensity.tolist() == [6.0]
    assert pair.long_intensity.tolist() == [108.0, 109.0]
