"""Memory-mapped reader for the lossless E1 RData conversion cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EICPair:
    short_intensity: NDArray[np.float32]
    short_retention_time: NDArray[np.float32]
    long_intensity: NDArray[np.float32]
    long_retention_time: NDArray[np.float32]


class EICFlatCache:
    """Read exact variable-length EIC pairs without materializing all cases."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        manifest = {}
        for line in (self.directory / "cache_manifest.txt").read_text().splitlines():
            key, value = line.split("=", 1)
            manifest[key] = value
        if manifest.get("format") != "hcrd-e1-flat-v1":
            raise ValueError("unsupported EIC cache format")
        self.sample_names = tuple(
            (self.directory / "sample_names.txt").read_text(encoding="utf-8").splitlines()
        )
        self.peak_names = tuple(
            (self.directory / "peak_names.txt").read_text(encoding="utf-8").splitlines()
        )
        self.sample_count = int(manifest["sample_count"])
        self.peak_count = int(manifest["peak_count"])
        self.case_count = int(manifest["case_count"])
        self.short_max_length = int(manifest["short_max_length"])
        self.long_max_length = int(manifest["long_max_length"])
        if len(self.sample_names) != self.sample_count:
            raise ValueError("sample-name count mismatch")
        if len(self.peak_names) != self.peak_count:
            raise ValueError("peak-name count mismatch")
        if self.case_count != self.sample_count * self.peak_count:
            raise ValueError("case count is not the sample/peak Cartesian product")
        self._short_intensity = self._map_float("short_intensity.f32", self.short_max_length)
        self._short_time = self._map_float(
            "short_retention_time.f32", self.short_max_length
        )
        self._long_intensity = self._map_float("long_intensity.f32", self.long_max_length)
        self._long_time = self._map_float(
            "long_retention_time.f32", self.long_max_length
        )
        self._short_length = self._map_length("short_length.i32")
        self._long_length = self._map_length("long_length.i32")

    def _map_float(self, name: str, width: int) -> NDArray[np.float32]:
        return np.memmap(
            self.directory / name,
            mode="r",
            dtype="<f4",
            shape=(self.case_count, width),
        )

    def _map_length(self, name: str) -> NDArray[np.int32]:
        return np.memmap(
            self.directory / name,
            mode="r",
            dtype="<i4",
            shape=(self.case_count,),
        )

    def flat_index(self, sample_index: int, peak_index: int) -> int:
        if not 0 <= sample_index < self.sample_count:
            raise IndexError("sample index out of bounds")
        if not 0 <= peak_index < self.peak_count:
            raise IndexError("peak index out of bounds")
        return sample_index * self.peak_count + peak_index

    def pair(self, sample_index: int, peak_index: int) -> EICPair:
        index = self.flat_index(sample_index, peak_index)
        short_length = int(self._short_length[index])
        long_length = int(self._long_length[index])
        return EICPair(
            short_intensity=np.asarray(self._short_intensity[index, :short_length]),
            short_retention_time=np.asarray(self._short_time[index, :short_length]),
            long_intensity=np.asarray(self._long_intensity[index, :long_length]),
            long_retention_time=np.asarray(self._long_time[index, :long_length]),
        )
