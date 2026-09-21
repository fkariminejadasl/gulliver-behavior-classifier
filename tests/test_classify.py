import pandas as pd
from conftest import ANIMAL, TAG, write_example

from gulliver.classify import OUTPUT_COLUMNS, classify_files, classify_to_file
from gulliver.model import CLASS_NAMES


def test_output_has_the_agreed_columns(tmp_path, classifier):
    output, report = classify_files(*write_example(tmp_path), classifier)

    assert list(output.columns) == OUTPUT_COLUMNS
    assert len(output) == report.accepted_bursts == 1
    assert output["tag-id"].tolist() == [TAG]
    assert output["animal-id"].tolist() == [ANIMAL]
    assert output["behaviour-classification"][0] in CLASS_NAMES
    assert output["behavior-according-to"].tolist() == ["machine"]


def test_three_bursts_of_one_fix_get_their_own_row(tmp_path, classifier):
    # 60 samples behind one GPS fix. The timestamp is the first sample of each
    # burst, so the three rows are one second apart.
    output, _ = classify_files(*write_example(tmp_path, sample_count=60), classifier)

    assert output["timestamp"].tolist() == [
        "2019-01-01 00:39:03.000",
        "2019-01-01 00:39:04.000",
        "2019-01-01 00:39:05.000",
    ]
    assert output["tag-id"].tolist() == [TAG] * 3


def test_no_usable_burst_writes_only_a_header(tmp_path, classifier):
    files = write_example(tmp_path, sample_count=19)
    output_file = tmp_path / "out.csv"

    report = classify_to_file(*files, output_file, classifier)

    assert report.accepted_bursts == 0
    written = pd.read_csv(output_file)
    assert list(written.columns) == OUTPUT_COLUMNS
    assert len(written) == 0


def test_written_file_matches_the_frame(tmp_path, classifier):
    files = write_example(tmp_path, sample_count=40)
    output_file = tmp_path / "out.csv"

    output, _ = classify_files(*files, classifier)
    classify_to_file(*files, output_file, classifier)

    pd.testing.assert_frame_equal(pd.read_csv(output_file), output)
