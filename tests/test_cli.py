from pathlib import Path

import pandas as pd
import pytest
from conftest import write_example

from gulliver.classify import OUTPUT_COLUMNS
from gulliver.cli import find_pairs, main, output_name


def test_find_pairs_matches_names(tmp_path):
    (tmp_path / "X-acceleration-2019.csv.gz").touch()
    (tmp_path / "X-gps-2019.csv.gz").touch()
    (tmp_path / "Y-gps-2020.csv").touch()

    pairs, lonely = find_pairs(tmp_path)

    assert pairs == [
        (tmp_path / "X-gps-2019.csv.gz", tmp_path / "X-acceleration-2019.csv.gz")
    ]
    assert lonely == [tmp_path / "Y-gps-2020.csv"]


@pytest.mark.parametrize(
    "gps_name",
    ["HG_TEXEL-gps-2019.csv.gz", "HG_TEXEL-gps-2019.csv"],
)
def test_output_name_drops_the_compression(gps_name):
    assert output_name(Path(gps_name)) == "HG_TEXEL-behaviour-2019.csv"


def test_runs_one_pair(tmp_path):
    gps_file, acceleration_file = write_example(tmp_path)
    output_file = tmp_path / "out.csv"

    code = main(
        [
            "--gps",
            str(gps_file),
            "--acceleration",
            str(acceleration_file),
            "--output",
            str(output_file),
        ]
    )

    assert code == 0
    assert list(pd.read_csv(output_file).columns) == OUTPUT_COLUMNS


def test_makes_the_output_folder(tmp_path):
    gps_file, acceleration_file = write_example(tmp_path)
    output_file = tmp_path / "new" / "out.csv"

    main(
        [
            "--gps",
            str(gps_file),
            "--acceleration",
            str(acceleration_file),
            "--output",
            str(output_file),
        ]
    )

    assert output_file.is_file()


def test_runs_every_pair_in_a_folder(tmp_path, capsys):
    exports = tmp_path / "exports"
    exports.mkdir()
    gps_file, acceleration_file = write_example(exports, sample_count=60)
    gps_file.rename(exports / "T-gps-2019.csv.gz")
    acceleration_file.rename(exports / "T-acceleration-2019.csv.gz")
    results = tmp_path / "results"

    assert main(["--directory", str(exports), "--output", str(results)]) == 0

    written = pd.read_csv(results / "T-behaviour-2019.csv")
    assert len(written) == 3
    assert "3 bursts classified" in capsys.readouterr().out


def test_needs_a_pair_or_a_folder(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "out.csv")])


def test_refuses_a_pair_and_a_folder_together(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--directory",
                str(tmp_path),
                "--gps",
                "a.csv",
                "--output",
                str(tmp_path / "out"),
            ]
        )
