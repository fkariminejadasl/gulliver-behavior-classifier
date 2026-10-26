"""Turn Movebank exports into one behaviour class per burst.

The output is the format agreed in https://github.com/inbo/bird-tracking/issues/297,
with two changes. The last two columns were renamed later, at the request of the
data publishers, to `behavioural-classification` and `study-specific-measurement`.
And `timestamp` is the timestamp of the acceleration sample that starts the
burst, not the timestamp of the GPS fix. One fix can carry up to ten
bursts, and each burst gets its own class, so the GPS timestamp cannot tell them
apart. The first burst of a fix starts on the fix itself, and the next ones
follow one second later each.
"""

from pathlib import Path

import pandas as pd
from torch import nn

from gulliver.model import classify
from gulliver.movebank import PreprocessingReport, load_movebank_data

OUTPUT_COLUMNS = [
    "tag-id",
    "animal-id",
    "timestamp",
    "behavioural-classification",
    "study-specific-measurement",
]
STUDY_SPECIFIC_MEASUREMENT = "machine"


def _movebank_time(times: pd.Series) -> pd.Series:
    """Format as the exports do: UTC, milliseconds, no zone suffix."""
    return times.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]


def classify_files(
    gps_file: str | Path,
    acceleration_file: str | Path,
    model: nn.Module | None = None,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Classify every burst in one pair of exports."""
    data = load_movebank_data(gps_file, acceleration_file)
    names, _ = classify(data.acceleration, data.ground_speed, model)

    output = pd.DataFrame(
        {
            "tag-id": data.metadata["tag-local-identifier"],
            "animal-id": data.metadata["individual-local-identifier"],
            "timestamp": _movebank_time(data.metadata["timestamp"]),
            "behavioural-classification": names,
        }
    )
    output["study-specific-measurement"] = STUDY_SPECIFIC_MEASUREMENT
    return output[OUTPUT_COLUMNS], data.report


def classify_to_file(
    gps_file: str | Path,
    acceleration_file: str | Path,
    output_file: str | Path,
    model: nn.Module | None = None,
) -> PreprocessingReport:
    """Classify one pair of exports and write the output CSV."""
    output, report = classify_files(gps_file, acceleration_file, model)
    output.to_csv(output_file, index=False)
    return report
