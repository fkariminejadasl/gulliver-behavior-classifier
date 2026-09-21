"""Read and validate Movebank GPS and acceleration exports.

One model input is a burst of 20 calibrated acceleration samples at 20 Hz,
matched to one GPS fix by tag, animal, and burst start timestamp. A longer
unbroken run is cut into as many bursts of 20 as it holds, so 60 clean samples
give 3 bursts; the leftover samples are dropped. Rows and bursts that fail a
check are skipped and counted, so a run says why data was lost. Accepted
bursts are clipped to the range the model was trained on.

The checks are the ones the model was trained behind, in bird-behavior
`behavior/data.py::identify_and_process_groups`, `::match_gps_to_groups` and
`scripts/data/bird_behavior_app_data.py::infer_save_app_data`.
"""

import gzip
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, fields
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

GPS_COLUMNS = [
    "timestamp",
    "ground-speed",
    "tag-local-identifier",
    "individual-local-identifier",
]
TILT_COLUMNS = ["tilt-x", "tilt-y", "tilt-z"]
ACCELERATION_COLUMNS = [
    "timestamp",
    "start-timestamp",
    *TILT_COLUMNS,
    "tag-local-identifier",
    "individual-local-identifier",
]
KEY_COLUMNS = [
    "tag-local-identifier",
    "individual-local-identifier",
    "start-timestamp",
]
BURST_COLUMNS = [*KEY_COLUMNS, "timestamp"]
BURST_LENGTH = 20
SAMPLE_PERIOD = pd.Timedelta(milliseconds=50)
MAX_SPEED = 30.0  # m/s. A faster GPS fix is an error, not a flying gull.
TILT_LIMIT = 2.0  # g. The range the model was trained on.
GZIP_MAGIC = b"\x1f\x8b"
MAX_GZIP_LAYERS = 4


@dataclass(frozen=True)
class PreprocessingReport:
    """Why rows and bursts were dropped. Row counts first, then burst counts."""

    acceleration_rows: int
    acceleration_groups: int
    missing_value_rows: int
    duplicate_time_rows: int
    short_run_rows: int
    partial_burst_rows: int
    bursts: int
    missing_gps_bursts: int
    conflicting_gps_bursts: int
    missing_speed_bursts: int
    fast_gps_bursts: int
    accepted_bursts: int
    clipped_values: int


@dataclass(frozen=True)
class MovebankData:
    """Validated model inputs and their Movebank identifiers."""

    acceleration: np.ndarray  # N x 20 x 3, clipped to +/- TILT_LIMIT
    ground_speed: np.ndarray  # N, metres per second
    metadata: pd.DataFrame  # N rows: the key columns, timestamp, burst-index
    report: PreprocessingReport


def _require_columns(data: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _open_csv(path: Path, stack: ExitStack) -> BinaryIO:
    """Open a CSV and peel off any gzip layers.

    Movebank downloads are sometimes gzipped twice, so the magic bytes decide
    how often to decompress, not the file name.
    """
    stream: BinaryIO = stack.enter_context(path.open("rb"))
    for _ in range(MAX_GZIP_LAYERS):
        if stream.peek(2)[:2] != GZIP_MAGIC:
            return stream
        stream = stack.enter_context(gzip.GzipFile(fileobj=stream))
    if stream.peek(2)[:2] == GZIP_MAGIC:
        raise ValueError(f"{path} is gzipped more than {MAX_GZIP_LAYERS} times")
    return stream


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    """Read the required columns of a CSV or gzipped CSV."""
    with ExitStack() as stack:
        header = pd.read_csv(
            _open_csv(path, stack), encoding="utf-8-sig", compression=None, nrows=0
        )
    _require_columns(header, required, path)
    with ExitStack() as stack:
        return pd.read_csv(
            _open_csv(path, stack),
            encoding="utf-8-sig",
            compression=None,
            usecols=required,
            low_memory=False,
        )


def _gps_speeds(gps: pd.DataFrame) -> pd.DataFrame:
    """One row per GPS fix: its ground speed and how many rows claimed it."""
    gps = gps.rename(columns={"timestamp": "start-timestamp"})
    gps["start-timestamp"] = pd.to_datetime(
        gps["start-timestamp"], utc=True, errors="coerce"
    )
    gps = gps[[*KEY_COLUMNS, "ground-speed"]].drop_duplicates()
    return gps.groupby(KEY_COLUMNS, dropna=False, as_index=False).agg(
        **{
            "ground-speed": ("ground-speed", "first"),
            "gps-rows": ("ground-speed", "size"),
        }
    )


def _cut_into_bursts(acceleration: pd.DataFrame, counts: Counter) -> pd.DataFrame:
    """Keep only the rows that fill a burst of consecutive samples.

    Rows are sorted per fix, a run ends wherever the step is not one sample
    period, and every run gives as many whole bursts as it holds.
    """
    acceleration = acceleration.sort_values([*KEY_COLUMNS, "timestamp"], kind="stable")

    repeated = acceleration.duplicated([*KEY_COLUMNS, "timestamp"])
    counts["duplicate_time_rows"] = int(repeated.sum())
    acceleration = acceleration[~repeated]

    keys = acceleration[KEY_COLUMNS]
    same_fix = ~(keys != keys.shift()).any(axis=1)
    in_step = acceleration["timestamp"].diff() == SAMPLE_PERIOD
    run_ids = (~(same_fix & in_step)).cumsum()

    position = acceleration.groupby(run_ids, sort=False).cumcount()
    run_sizes = acceleration.groupby(run_ids, sort=False)["timestamp"].transform("size")
    short = run_sizes < BURST_LENGTH
    fits = position < (run_sizes // BURST_LENGTH) * BURST_LENGTH
    counts["short_run_rows"] = int(short.sum())
    counts["partial_burst_rows"] = int((~fits & ~short).sum())
    return acceleration[fits]


def load_movebank_data(
    gps_file: str | Path, acceleration_file: str | Path
) -> MovebankData:
    """Load model-ready bursts from Movebank CSV or gzipped CSV files."""
    gps_file = Path(gps_file)
    acceleration_file = Path(acceleration_file)
    gps = _read_csv(gps_file, GPS_COLUMNS)
    acceleration = _read_csv(acceleration_file, ACCELERATION_COLUMNS)

    counts = Counter({"acceleration_rows": len(acceleration)})

    for column in ("timestamp", "start-timestamp"):
        acceleration[column] = pd.to_datetime(
            acceleration[column], utc=True, errors="coerce"
        )
    incomplete = acceleration[["timestamp", *KEY_COLUMNS, *TILT_COLUMNS]].isna()
    incomplete = incomplete.any(axis=1)
    counts["missing_value_rows"] = int(incomplete.sum())
    acceleration = acceleration[~incomplete]
    counts["acceleration_groups"] = acceleration.groupby(
        KEY_COLUMNS, dropna=False, sort=False
    ).ngroups

    if len(acceleration):
        acceleration = _cut_into_bursts(acceleration, counts)
    samples = acceleration[TILT_COLUMNS].to_numpy(dtype=np.float32)
    samples = samples.reshape(-1, BURST_LENGTH, len(TILT_COLUMNS))
    counts["bursts"] = len(samples)

    # One row per burst: its fix, when it starts, and its place among the
    # bursts of that fix. `timestamp` is the first sample of the burst, so the
    # bursts of one fix stay apart; only the first one equals `start-timestamp`.
    starts = acceleration.iloc[::BURST_LENGTH][BURST_COLUMNS].reset_index(drop=True)
    starts["burst-index"] = starts.groupby(KEY_COLUMNS, sort=False).cumcount()
    bursts = starts.merge(_gps_speeds(gps), on=KEY_COLUMNS, how="left")

    no_gps = bursts["gps-rows"].isna()
    conflicting = bursts["gps-rows"] > 1  # one fix, two different speeds
    matched = ~no_gps & ~conflicting
    no_speed = matched & bursts["ground-speed"].isna()
    # 30 m/s and above is a broken fix, not a flying gull.
    too_fast = matched & (bursts["ground-speed"] >= MAX_SPEED)
    counts["missing_gps_bursts"] = int(no_gps.sum())
    counts["conflicting_gps_bursts"] = int(conflicting.sum())
    counts["missing_speed_bursts"] = int(no_speed.sum())
    counts["fast_gps_bursts"] = int(too_fast.sum())

    keep = matched & ~no_speed & ~too_fast
    samples = samples[keep.to_numpy()]
    bursts = bursts[keep]
    counts["accepted_bursts"] = len(samples)
    counts["clipped_values"] = int((np.abs(samples) > TILT_LIMIT).sum())
    np.clip(samples, -TILT_LIMIT, TILT_LIMIT, out=samples)

    return MovebankData(
        acceleration=samples,
        ground_speed=bursts["ground-speed"].to_numpy(dtype=np.float32),
        metadata=bursts[[*BURST_COLUMNS, "burst-index"]].reset_index(drop=True),
        report=PreprocessingReport(
            **{
                field.name: int(counts[field.name])
                for field in fields(PreprocessingReport)
            }
        ),
    )
