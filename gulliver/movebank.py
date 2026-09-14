"""Read and validate Movebank GPS and acceleration exports.

One model input is a burst of exactly 20 calibrated acceleration samples at
20 Hz, matched to one GPS fix by tag, animal, and burst start timestamp.
Incomplete, irregular, duplicated, or unmatched bursts are skipped and counted.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GPS_COLUMNS = {
    "timestamp",
    "ground-speed",
    "tag-local-identifier",
    "individual-local-identifier",
}
ACCELERATION_COLUMNS = {
    "timestamp",
    "start-timestamp",
    "tilt-x",
    "tilt-y",
    "tilt-z",
    "tag-local-identifier",
    "individual-local-identifier",
}
KEY_COLUMNS = [
    "tag-local-identifier",
    "individual-local-identifier",
    "start-timestamp",
]
BURST_LENGTH = 20
SAMPLE_PERIOD = pd.Timedelta(milliseconds=50)


@dataclass(frozen=True)
class PreprocessingReport:
    acceleration_groups: int
    accepted_bursts: int
    incomplete_bursts: int
    duplicate_sample_bursts: int
    irregular_sample_bursts: int
    missing_acceleration_bursts: int
    missing_gps_bursts: int
    duplicate_gps_bursts: int
    missing_speed_bursts: int


@dataclass(frozen=True)
class MovebankData:
    """Validated model inputs and their Movebank identifiers."""

    acceleration: np.ndarray
    ground_speed: np.ndarray
    metadata: pd.DataFrame
    report: PreprocessingReport


def _require_columns(data: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    _require_columns(data, required, path)
    return data


def load_movebank_data(
    gps_file: str | Path, acceleration_file: str | Path
) -> MovebankData:
    """Load complete Movebank bursts from CSV or gzipped CSV files."""
    gps_file = Path(gps_file)
    acceleration_file = Path(acceleration_file)
    gps = _read_csv(gps_file, GPS_COLUMNS)
    acceleration = _read_csv(acceleration_file, ACCELERATION_COLUMNS)

    gps = gps.rename(columns={"timestamp": "start-timestamp"})
    for frame in (gps, acceleration):
        frame["start-timestamp"] = pd.to_datetime(
            frame["start-timestamp"], utc=True, errors="coerce"
        )
    acceleration["timestamp"] = pd.to_datetime(
        acceleration["timestamp"], utc=True, errors="coerce"
    )

    gps_groups = {
        key: group for key, group in gps.groupby(KEY_COLUMNS, dropna=False, sort=False)
    }
    arrays = []
    speeds = []
    metadata = []
    counts = {
        "acceleration_groups": 0,
        "accepted_bursts": 0,
        "incomplete_bursts": 0,
        "duplicate_sample_bursts": 0,
        "irregular_sample_bursts": 0,
        "missing_acceleration_bursts": 0,
        "missing_gps_bursts": 0,
        "duplicate_gps_bursts": 0,
        "missing_speed_bursts": 0,
    }

    for key, burst in acceleration.groupby(KEY_COLUMNS, dropna=False, sort=False):
        counts["acceleration_groups"] += 1
        if len(burst) != BURST_LENGTH:
            counts["incomplete_bursts"] += 1
            continue
        if burst[["timestamp", "tilt-x", "tilt-y", "tilt-z"]].isna().any().any():
            counts["missing_acceleration_bursts"] += 1
            continue
        if burst["timestamp"].duplicated().any():
            counts["duplicate_sample_bursts"] += 1
            continue

        burst = burst.sort_values("timestamp")
        if not (burst["timestamp"].diff().iloc[1:] == SAMPLE_PERIOD).all():
            counts["irregular_sample_bursts"] += 1
            continue

        matching_gps = gps_groups.get(key)
        if matching_gps is None:
            counts["missing_gps_bursts"] += 1
            continue
        if len(matching_gps) != 1:
            counts["duplicate_gps_bursts"] += 1
            continue
        speed = matching_gps.iloc[0]["ground-speed"]
        if pd.isna(speed):
            counts["missing_speed_bursts"] += 1
            continue

        arrays.append(burst[["tilt-x", "tilt-y", "tilt-z"]].to_numpy(dtype=np.float32))
        speeds.append(float(speed))
        metadata.append(key)
        counts["accepted_bursts"] += 1

    acceleration_array = (
        np.stack(arrays) if arrays else np.empty((0, 20, 3), dtype=np.float32)
    )
    metadata_frame = pd.DataFrame(metadata, columns=KEY_COLUMNS)
    return MovebankData(
        acceleration=acceleration_array,
        ground_speed=np.asarray(speeds, dtype=np.float32),
        metadata=metadata_frame,
        report=PreprocessingReport(**counts),
    )
