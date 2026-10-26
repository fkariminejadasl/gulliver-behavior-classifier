# GULLIVER behaviour classifier

Classify gull behaviour from Movebank GPS and acceleration exports. Each burst of 20 acceleration samples, with the ground speed of its GPS fix, gets one of 9 behaviour classes from a small trained convolutional neural network.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/fkariminejadasl/gulliver-behavior-classifier
```

The classifier is small and runs on the processor, so the CPU build of PyTorch
is enough. The trained weights ship inside the package; nothing else to
download.

## Use

One pair of files:

```bash
gulliver-classify \
  --gps HG_TEXEL-gps-2019.csv.gz \
  --acceleration HG_TEXEL-acceleration-2019.csv.gz \
  --output HG_TEXEL-behaviour-2019.csv
```

Every pair in a folder. Each file whose name holds `gps` is paired with the file
of the same name holding `acceleration` instead, and the output is named after
it with `behaviour`, so `HG_TEXEL-gps-2019.csv.gz` gives
`HG_TEXEL-behaviour-2019.csv`:

```bash
gulliver-classify --directory exports --output results
```

Each file prints how many bursts were classified and what was dropped.

## Input

Two CSV files, plain or gzipped. Movebank downloads are sometimes gzipped
twice; that is handled, so no unpacking by hand.

| file | columns used |
|---|---|
| GPS | `timestamp`, `ground-speed`, `tag-local-identifier`, `individual-local-identifier` |
| acceleration | `timestamp`, `start-timestamp`, `tilt-x`, `tilt-y`, `tilt-z`, `tag-local-identifier`, `individual-local-identifier` |

The acceleration `start-timestamp` joins with the GPS `timestamp`. Other columns
are ignored.

## Output

One CSV, one row per classified burst.

| column | value |
|---|---|
| `tag-id` | from `tag-local-identifier` |
| `animal-id` | from `individual-local-identifier` |
| `timestamp` | the acceleration sample that starts the burst |
| `behavioural-classification` | one of the 9 classes below |
| `study-specific-measurement` | always `machine` |

The classes are `flap`, `exflap`, `soar`, `boat`, `float`, `stationary`,
`walk`, `mixed`, `peck`: the classes of Shamoun-Baranes et al. (2016), in lower
case. [METHODS.md](METHODS.md) says what each one means.

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
