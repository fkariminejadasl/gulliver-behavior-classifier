"""Classify behaviour in Movebank exports from the command line."""

import argparse
import sys
from pathlib import Path

from gulliver.classify import classify_to_file
from gulliver.model import CHECKPOINT, load_classifier
from gulliver.movebank import PreprocessingReport

GPS_MARK = "gps"
ACCELERATION_MARK = "acceleration"
OUTPUT_MARK = "behaviour"

USAGE = """\
One pair of files:
  gulliver-classify --gps GPS.csv.gz --acceleration ACCELERATION.csv.gz \
--output OUT.csv

Every pair in a folder:
  gulliver-classify --directory EXPORTS --output RESULTS
"""


def find_pairs(directory: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Pair every GPS export with the acceleration export of the same name.

    Returns the pairs, and the GPS exports that have no partner.
    """
    pairs = []
    lonely = []
    for gps_file in sorted(directory.iterdir()):
        if not gps_file.is_file() or GPS_MARK not in gps_file.name:
            continue
        partner = gps_file.with_name(gps_file.name.replace(GPS_MARK, ACCELERATION_MARK))
        if partner.is_file():
            pairs.append((gps_file, partner))
        else:
            lonely.append(gps_file)
    return pairs, lonely


def output_name(gps_file: Path) -> str:
    """HG_TEXEL-gps-2019.csv.gz becomes HG_TEXEL-behaviour-2019.csv."""
    name = gps_file.name.replace(GPS_MARK, OUTPUT_MARK)
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    return name


def report_lines(report: PreprocessingReport) -> list[str]:
    """The counts worth reading: the result, then whatever was lost."""
    lines = [
        f"{report.accepted_bursts} bursts classified, "
        f"from {report.acceleration_rows} acceleration rows"
    ]
    dropped = {
        "rows with a missing value": report.missing_value_rows,
        "repeated timestamps": report.duplicate_time_rows,
        "rows in runs shorter than a burst": report.short_run_rows,
        "rows left over after the last whole burst": report.partial_burst_rows,
        "bursts with no GPS fix": report.missing_gps_bursts,
        "bursts whose fix carries two speeds": report.conflicting_gps_bursts,
        "bursts with no GPS speed": report.missing_speed_bursts,
        "bursts with an impossible GPS speed": report.fast_gps_bursts,
    }
    lines += [f"dropped {count} {what}" for what, count in dropped.items() if count]
    if report.clipped_values:
        lines.append(f"clipped {report.clipped_values} tilt values")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gulliver-classify",
        description="Classify gull behaviour in Movebank GPS and acceleration "
        "exports. Both may be plain .csv or gzipped.",
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gps", type=Path, help="one GPS export")
    parser.add_argument(
        "--acceleration", type=Path, help="the matching acceleration export"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="a folder of exports. Every file whose name holds gps is paired "
        "with the file of the same name holding acceleration instead.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="the output CSV, or with --directory the folder to write them into",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="trained weights (default: the ones shipped with this package)",
    )
    args = parser.parse_args(argv)

    if args.directory is not None and (args.gps or args.acceleration):
        parser.error("give either --directory or --gps with --acceleration, not both")
    if args.directory is None and not (args.gps and args.acceleration):
        parser.error("give --gps with --acceleration, or --directory")

    if args.directory is not None:
        pairs, lonely = find_pairs(args.directory)
        for gps_file in lonely:
            print(f"skipped {gps_file.name}: no matching acceleration export")
        if not pairs:
            parser.error(f"no pair of exports found in {args.directory}")
        args.output.mkdir(parents=True, exist_ok=True)
        jobs = [(gps, acc, args.output / output_name(gps)) for gps, acc in pairs]
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(args.gps, args.acceleration, args.output)]

    model = load_classifier(args.checkpoint)
    for gps_file, acceleration_file, output_file in jobs:
        print(f"{gps_file.name} + {acceleration_file.name} -> {output_file}")
        report = classify_to_file(gps_file, acceleration_file, output_file, model)
        for line in report_lines(report):
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
