# GULLIVER behaviour classifier

This repository will classify gull behaviour from Movebank GPS and acceleration
exports with the exp197 model. It is being built in small, reviewable steps.

## Current step: Movebank preprocessing

`gulliver.movebank.load_movebank_data` reads the GPS and acceleration exports
and returns the bursts the model can be given. A burst is 20 calibrated
acceleration samples 50 milliseconds apart, from one tag, one animal and one
`start-timestamp`, together with the ground speed of the matching GPS fix.

Both files may be plain `.csv` or gzipped. Movebank downloads are sometimes
gzipped twice; the reader looks at the file's first bytes and decompresses as
often as needed, so those files need no unpacking by hand.

What is kept, in order:

1. rows with a missing timestamp, tag, animal or tilt value are dropped;
2. a repeated timestamp within one fix is dropped;
3. the remaining samples are cut into runs of consecutive 50 millisecond
   steps, and each run gives as many whole bursts of 20 as it holds, so 60
   clean samples give 3 bursts and 65 give 3 bursts with 5 samples left over;
4. a burst is dropped when its GPS fix is missing, when one fix carries two
   different speeds, when the speed is missing, or when the speed is 30 metres
   per second or more;
5. the tilt values of the kept bursts are clipped to between -2 and 2 g.

These are the checks the model was trained behind. The returned
`PreprocessingReport` counts every dropped row and burst by reason, so a file
that yields little data says why.

On the 2019 Texel herring gull export (1 505 524 acceleration rows), this
keeps 74 118 bursts and takes about 6 seconds.

## Development check

```bash
python -m pytest
```

## License

MIT
