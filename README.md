# GULLIVER behaviour classifier

This repository will classify gull behaviour from Movebank GPS and acceleration
exports with the exp197 model. It is being built in small, reviewable steps.

## Current step: Movebank preprocessing

`gulliver.movebank.load_movebank_data` reads `.csv` or `.csv.gz` files. It:

- groups acceleration rows by tag, animal, and `start-timestamp`;
- requires 20 unique samples spaced by 50 milliseconds;
- joins one GPS fix with the same tag, animal, and timestamp;
- uses calibrated `tilt-x`, `tilt-y`, and `tilt-z` values;
- requires `ground-speed` in metres per second; and
- returns accepted bursts plus counts for every skipped group.

Model inference and the command-line interface will be added in the next step.

## Development check

```bash
python -m pytest
```

## License

MIT
