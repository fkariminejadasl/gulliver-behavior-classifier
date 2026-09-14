import pandas as pd
import pytest

from gulliver.movebank import load_movebank_data


def write_example(tmp_path, sample_count=20, sample_period="50ms", speed=3.5):
    start = pd.Timestamp("2019-01-01 00:39:03")
    gps = pd.DataFrame(
        {
            "timestamp": [start],
            "ground-speed": [speed],
            "tag-local-identifier": [6073],
            "individual-local-identifier": [5460390],
        }
    )
    acceleration = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=sample_count, freq=sample_period),
            "start-timestamp": [start] * sample_count,
            "tilt-x": range(sample_count),
            "tilt-y": range(sample_count),
            "tilt-z": range(sample_count),
            "tag-local-identifier": [6073] * sample_count,
            "individual-local-identifier": [5460390] * sample_count,
        }
    )
    gps_file = tmp_path / "gps.csv.gz"
    acceleration_file = tmp_path / "acceleration.csv.gz"
    gps.to_csv(gps_file, index=False)
    acceleration.to_csv(acceleration_file, index=False)
    return gps_file, acceleration_file


def test_loads_complete_gzipped_burst(tmp_path):
    gps_file, acceleration_file = write_example(tmp_path)

    data = load_movebank_data(gps_file, acceleration_file)

    assert data.acceleration.shape == (1, 20, 3)
    assert data.ground_speed.tolist() == [3.5]
    assert data.metadata.iloc[0]["tag-local-identifier"] == 6073
    assert data.report.accepted_bursts == 1


@pytest.mark.parametrize(
    ("sample_count", "sample_period", "report_field"),
    [(19, "50ms", "incomplete_bursts"), (20, "51ms", "irregular_sample_bursts")],
)
def test_skips_invalid_burst(tmp_path, sample_count, sample_period, report_field):
    gps_file, acceleration_file = write_example(
        tmp_path, sample_count=sample_count, sample_period=sample_period
    )

    data = load_movebank_data(gps_file, acceleration_file)

    assert data.acceleration.shape == (0, 20, 3)
    assert getattr(data.report, report_field) == 1


def test_missing_column_has_clear_error(tmp_path):
    gps = tmp_path / "gps.csv"
    pd.DataFrame({"timestamp": []}).to_csv(gps, index=False)
    _, acceleration = write_example(tmp_path)

    with pytest.raises(ValueError, match="ground-speed"):
        load_movebank_data(gps, acceleration)
