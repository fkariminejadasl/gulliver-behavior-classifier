# GULLIVER behaviour classifier

Classify gull behaviour from Movebank GPS and acceleration exports. Each burst of 20 acceleration samples, with the ground speed of its GPS fix, gets one of 9 behaviour classes from a small trained convolutional neural network.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/fkariminejadasl/gulliver-behavior-classifier
```

The classifier is small and runs on the processor, so the CPU build of PyTorch is enough. The trained weights ship with the package, so nothing else needs to be downloaded.

## Usage

This example uses gull GPS and acceleration data from [Shamoun-Baranes et al. 2026](https://doi.org/10.5281/zenodo.22297895):

- [HG_TEXEL-gps-2019.csv.gz](https://zenodo.org/records/22297895/files/HG_TEXEL-gps-2019.csv.gz?download=1)
- [HG_TEXEL-acceleration-2019.csv.gz](https://zenodo.org/records/22297895/files/HG_TEXEL-acceleration-2019.csv.gz?download=1)

To classify, either provide the input and output **file names**:

```bash
gulliver-classify \
  --gps HG_TEXEL-gps-2019.csv.gz \
  --acceleration HG_TEXEL-acceleration-2019.csv.gz \
  --output HG_TEXEL-behaviour-2019.csv
```

Or the input and output **directories**:

```
gulliver-classify \
  --directory . \
  --output output
```

If directories are provided, files with names containing `gps` and `acceleration` will be paired. E.g. `file-gps-1.csv` and `file-acceleration-1.csv` -> `output/file-behaviour-1.csv`.

The classifier will print how many bursts were classified and what was dropped.

## Input data

Input data can be provided as `csv` or gzipped `csv.gz` and are expected to have the following columns (other columns are ignored):

GPS:

- `timestamp`
- `ground-speed`
- `tag-local-identifier`
- `individual-local-identifier`

Acceleration:

- `timestamp`
- `start-timestamp`
- `tilt-x`
- `tilt-y`
- `tilt-z`
- `tag-local-identifier`
- `individual-local-identifier`

These columns follow the [Movebank Attribute Dictionary](http://vocab.nerc.ac.uk/collection/MVB). GPS and accelleration data are joined on `tag-local-identifier`, `individual-local-identifier` and `gps.timestamp = acceleration.start-timestamp`.

## Output data

Output data is a CSV file with the following columns:

- `tag-id`: from `tag-local-identifier`
- `animal-id`: from `individual-local-identifier`
- `timestamp`: the acceleration sample that starts the burst
- `behavioural-classification`: one of the 9 classes below
- `study-specific-measurement`: always `machine`

The classes are `flap`, `exflap`, `soar`, `boat`, `float`, `stationary`, `walk`, `mixed`, `peck`: the classes of Shamoun-Baranes et al. (2016), in lowercase. See [METHODS.md](METHODS.md) for definitions.

`timestamp` is the burst's own time, not the GPS fix's. One fix can carry up to
ten bursts and each gets its own class, so the GPS timestamp cannot keep them
apart. The first burst of a fix starts on the fix itself and the next ones
follow a second later each.

## What a burst is, and what is skipped

A burst is 20 acceleration samples 50 milliseconds apart, from one tag, one
animal and one `start-timestamp`, with the ground speed of the matching GPS fix.
In order:

1. rows with a missing timestamp, tag, animal or tilt value are dropped;
2. a repeated timestamp within one fix is dropped;
3. the remaining samples are cut into runs of consecutive 50 millisecond steps,
   and each run gives as many whole bursts of 20 as it holds, so 60 clean
   samples give 3 bursts and 65 give 3 bursts with 5 samples left over;
4. a burst is dropped when its GPS fix is missing, when one fix carries two
   different speeds, when the speed is missing, or when the speed is 30 metres
   per second or more;
5. the tilt values of the kept bursts are clipped to between -2 and 2 g.

These are the checks the model was trained behind. Every dropped row and burst
is counted by reason, so a file that yields little data says why.

## The model

A 5429-parameter dilated convolutional network, trained with
[bird-behavior](https://github.com/fkariminejadasl/bird-behavior), in
`gulliver/models/197_best.pth` (80 KB). It reads 7 channels: the three
acceleration axes, the scaled GPS speed, and three magnitude channels derived
from the axes. How it was trained, on which data,
and how well it does: [METHODS.md](METHODS.md).

On the 2019 Texel herring gull export, 1 505 524 acceleration rows give 74 118
classified bursts in about 11 seconds.

## Development check

```bash
pip install .[test]
python -m pytest
```

## Cite

Karimi Nejadasl F, Rakhimberdiev E. GULLIVER behaviour classifier. Zenodo.
<https://doi.org/10.5281/zenodo.23666749>

This DOI (Digital Object Identifier) always leads to the newest version.

## License

MIT
