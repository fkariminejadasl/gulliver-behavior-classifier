"""Small Movebank exports to test against, built in memory."""

import gzip

import numpy as np
import pandas as pd
import pytest

from gulliver.model import load_classifier

START = pd.Timestamp("2019-01-01 00:39:03")
TAG = 6073
ANIMAL = 5460390


@pytest.fixture(scope="session")
def classifier():
    return load_classifier()


def make_frames(sample_count=20, sample_period="50ms", speed=3.5):
    """One GPS fix and `sample_count` acceleration samples behind it."""
    gps = pd.DataFrame(
        {
            "timestamp": [START],
            "ground-speed": [speed],
            "tag-local-identifier": [TAG],
            "individual-local-identifier": [ANIMAL],
        }
    )
    tilt = np.arange(sample_count) / 100
    acceleration = pd.DataFrame(
        {
            "timestamp": pd.date_range(START, periods=sample_count, freq=sample_period),
            "start-timestamp": [START] * sample_count,
            "tilt-x": tilt,
            "tilt-y": tilt,
            "tilt-z": tilt,
            "tag-local-identifier": [TAG] * sample_count,
            "individual-local-identifier": [ANIMAL] * sample_count,
        }
    )
    return gps, acceleration


def write_frames(tmp_path, gps, acceleration, gzip_layers=1):
    files = []
    for name, frame in (("gps", gps), ("acceleration", acceleration)):
        payload = frame.to_csv(index=False).encode()
        for _ in range(gzip_layers):
            payload = gzip.compress(payload)
        path = tmp_path / f"{name}.csv{'.gz' if gzip_layers else ''}"
        path.write_bytes(payload)
        files.append(path)
    return files


def write_example(tmp_path, gzip_layers=1, **kwargs):
    return write_frames(tmp_path, *make_frames(**kwargs), gzip_layers=gzip_layers)
