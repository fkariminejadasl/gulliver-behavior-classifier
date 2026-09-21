import numpy as np
import pandas as pd
import pytest
from conftest import TAG, make_frames, write_example, write_frames

from gulliver.movebank import load_movebank_data


@pytest.mark.parametrize("gzip_layers", [0, 1, 2])
def test_reads_plain_and_gzipped_files(tmp_path, gzip_layers):
    # Movebank downloads are sometimes gzipped twice.
    data = load_movebank_data(*write_example(tmp_path, gzip_layers=gzip_layers))

    assert data.acceleration.shape == (1, 20, 3)
    assert data.ground_speed.tolist() == [3.5]
    assert data.metadata.iloc[0]["tag-local-identifier"] == TAG
    assert data.report.accepted_bursts == 1


def test_long_run_becomes_several_bursts(tmp_path):
    data = load_movebank_data(*write_example(tmp_path, sample_count=65))

    assert data.acceleration.shape == (3, 20, 3)
    assert data.metadata["burst-index"].tolist() == [0, 1, 2]
    assert data.ground_speed.tolist() == [3.5, 3.5, 3.5]
    assert data.report.partial_burst_rows == 5


def test_gap_splits_the_run(tmp_path):
    gps, acceleration = make_frames(sample_count=41)
    acceleration.loc[20:, "timestamp"] += pd.Timedelta(milliseconds=50)

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.acceleration.shape == (2, 20, 3)
    assert data.report.partial_burst_rows == 1


@pytest.mark.parametrize(
    ("sample_count", "sample_period", "short_rows"),
    [(19, "50ms", 19), (20, "51ms", 20)],
)
def test_skips_runs_shorter_than_a_burst(
    tmp_path, sample_count, sample_period, short_rows
):
    data = load_movebank_data(
        *write_example(tmp_path, sample_count=sample_count, sample_period=sample_period)
    )

    assert data.acceleration.shape == (0, 20, 3)
    assert data.report.short_run_rows == short_rows


def test_missing_sample_breaks_the_run(tmp_path):
    gps, acceleration = make_frames(sample_count=21)
    acceleration.loc[10, "tilt-y"] = np.nan

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.report.missing_value_rows == 1
    assert data.report.short_run_rows == 20  # two runs of 10, neither long enough
    assert data.acceleration.shape == (0, 20, 3)


def test_drops_repeated_timestamps(tmp_path):
    gps, acceleration = make_frames(sample_count=21)
    acceleration.loc[20, "timestamp"] = acceleration.loc[19, "timestamp"]

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.report.duplicate_time_rows == 1
    assert data.acceleration.shape == (1, 20, 3)


def test_skips_burst_without_a_gps_fix(tmp_path):
    gps, acceleration = make_frames()
    gps["timestamp"] += pd.Timedelta(seconds=1)

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.report.missing_gps_bursts == 1
    assert data.report.accepted_bursts == 0


def test_skips_burst_with_two_speeds_for_one_fix(tmp_path):
    gps, acceleration = make_frames()
    gps = pd.concat([gps, gps.assign(**{"ground-speed": 9.0})], ignore_index=True)

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.report.conflicting_gps_bursts == 1


def test_repeated_gps_row_is_not_a_conflict(tmp_path):
    gps, acceleration = make_frames()
    gps = pd.concat([gps, gps], ignore_index=True)

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.report.accepted_bursts == 1


def test_skips_burst_without_speed(tmp_path):
    data = load_movebank_data(*write_example(tmp_path, speed=np.nan))

    assert data.report.missing_speed_bursts == 1
    assert data.report.accepted_bursts == 0


def test_skips_impossible_speed(tmp_path):
    # 30 m/s and above is a broken fix, not a flying gull.
    data = load_movebank_data(*write_example(tmp_path, speed=30.0))

    assert data.report.fast_gps_bursts == 1
    assert data.report.accepted_bursts == 0


def test_clips_tilt_to_the_trained_range(tmp_path):
    gps, acceleration = make_frames()
    acceleration.loc[0, "tilt-x"] = 5.0
    acceleration.loc[1, "tilt-y"] = -5.0

    data = load_movebank_data(*write_frames(tmp_path, gps, acceleration))

    assert data.report.clipped_values == 2
    assert data.acceleration.max() == 2.0
    assert data.acceleration.min() == -2.0


def test_missing_column_has_clear_error(tmp_path):
    gps = tmp_path / "gps.csv"
    pd.DataFrame({"timestamp": []}).to_csv(gps, index=False)
    _, acceleration = write_example(tmp_path)

    with pytest.raises(ValueError, match="ground-speed"):
        load_movebank_data(gps, acceleration)
