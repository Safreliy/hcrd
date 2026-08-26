"""Development-only pilot on one unlocked CRCNS HC-1 recording."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
DATA_ROOT = REPOSITORY_ROOT / "third_party" / "crcns_hc1"
OUTPUT = ROOT / "results" / "crcns_hc1_d4_development"

sys.path.insert(0, str(ROOT / "src"))

from hcrd import (  # noqa: E402
    classical_compact_impulse_scores,
    event_average_precision,
    hcrd_area_anomaly_score,
    hcrd_concentration_anomaly_score,
    intracellular_spike_times,
    robust_multichannel_max,
    spike_bandpass,
)


def chunked_score(
    signal: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    *,
    core_samples: int,
    padding_samples: int,
) -> np.ndarray:
    """Apply a noncausal score in padded chunks and retain only each core."""

    output = np.empty(signal.size, dtype=float)
    for start in range(0, signal.size, core_samples):
        stop = min(signal.size, start + core_samples)
        left = max(0, start - padding_samples)
        right = min(signal.size, stop + padding_samples)
        local = np.asarray(scorer(signal[left:right]), dtype=float)
        if local.shape != (right - left,):
            raise RuntimeError("invalid local score shape")
        output[start:stop] = local[start - left : stop - left]
    return output


def load_recording(root: str, stem: str, duration_seconds: float | None):
    dat = next((DATA_ROOT / root).rglob(f"{stem}.dat"))
    xml = dat.with_suffix(".xml")
    parameters = ET.parse(xml).getroot()
    acquisition = parameters.find("acquisitionSystem")
    if acquisition is None:
        raise RuntimeError("missing acquisition metadata")
    channels = int(acquisition.findtext("nChannels", "0"))
    sampling_rate = float(acquisition.findtext("samplingRate", "0"))
    raw = np.memmap(dat, dtype="<i2", mode="r").reshape(-1, channels)
    stop = raw.shape[0]
    if duration_seconds is not None:
        stop = min(stop, int(round(duration_seconds * sampling_rate)))
    # Official paired HC-1 exports use the first five raw channels as the
    # extracellular array and channel 5 as the independent intracellular trace
    # for this development recording.
    return np.asarray(raw[:stop, :5], dtype=float), np.asarray(raw[:stop, 5], dtype=float), sampling_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    args = parser.parse_args()
    root = "d5331"
    stem = "d533101"
    extracellular, intracellular, sampling_rate = load_recording(
        root, stem, args.duration_seconds
    )
    truth = intracellular_spike_times(
        intracellular,
        sampling_rate=sampling_rate,
    )
    filtered = np.column_stack(
        [
            spike_bandpass(extracellular[:, channel], sampling_rate=sampling_rate)
            for channel in range(extracellular.shape[1])
        ]
    )
    channel_scores: dict[str, list[np.ndarray]] = {
        "amplitude": [],
        "neo": [],
        "cwt_mexh": [],
        "hcrd_L8_max": [],
        "hcrd_concentration_L8_max": [],
    }
    core = int(round(2.0 * sampling_rate))
    padding = int(round(0.05 * sampling_rate))
    for channel in range(filtered.shape[1]):
        values = filtered[:, channel]
        classical = classical_compact_impulse_scores(values)
        channel_scores["amplitude"].append(classical["amplitude"])
        channel_scores["neo"].append(classical["neo"])
        coefficients, _ = pywt.cwt(values, [1, 2, 3, 4, 6, 8], "mexh")
        channel_scores["cwt_mexh"].append(np.max(np.abs(coefficients), axis=0))
        channel_scores["hcrd_L8_max"].append(
            chunked_score(
                values,
                lambda item: hcrd_area_anomaly_score(
                    item, max_levels=8, aggregation="max"
                ),
                core_samples=core,
                padding_samples=padding,
            )
        )
        channel_scores["hcrd_concentration_L8_max"].append(
            chunked_score(
                values,
                lambda item: hcrd_concentration_anomaly_score(
                    item, max_levels=8
                ),
                core_samples=core,
                padding_samples=padding,
            )
        )
    results = []
    tolerance = int(round(0.001 * sampling_rate))
    refractory = int(round(0.001 * sampling_rate))
    for method, rows in channel_scores.items():
        fused = robust_multichannel_max(np.vstack(rows))
        results.append(
            {
                "method": method,
                "event_average_precision": event_average_precision(
                    fused,
                    truth,
                    tolerance_samples=tolerance,
                    refractory_samples=refractory,
                ),
            }
        )
    payload = {
        "status": "development_only",
        "recording": f"{root}_{stem}",
        "duration_seconds": extracellular.shape[0] / sampling_rate,
        "sampling_rate": sampling_rate,
        "ground_truth_events": int(truth.size),
        "tolerance_seconds": tolerance / sampling_rate,
        "results": results,
        "confirmation_accessed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "pilot_d533101.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
