# Data and model

This classifier is for the gull tracking data published through
[inbo/bird-tracking](https://github.com/inbo/bird-tracking). It reads their GPS
and acceleration files, applies the rules under Data, and classifies each burst
with the model under Model.

The model is already trained, and its weights ship with this repository.
Training and its labeled data are described under Training, for reference:
they say where the model comes from and how far to trust it, but nothing needs
to be trained to use it. The README says how to run the code.

## Data

### Where the data come from

- The input is the GPS (Global Positioning System) and acceleration data of
  gull studies with UvA-BiTS tags (University of Amsterdam Bird Tracking
  System), published as CSV files in the Movebank format. The studies are
  listed in [inbo/bird-tracking](https://github.com/inbo/bird-tracking).
- Example: HG_TEXEL, herring gulls breeding on Texel, DOI
  [10.5281/zenodo.20706394](https://doi.org/10.5281/zenodo.20706394),
  downloaded from <https://zenodo.org/records/22297895>. It has one GPS file
  and one acceleration file per year: `HG_TEXEL-gps-<year>.csv.gz` and
  `HG_TEXEL-acceleration-<year>.csv.gz`.
- Acceleration is used in g, as published (`tilt-x`, `tilt-y`, `tilt-z`). The
  raw sensor counts (`acceleration-raw-x`, ...) are not used.
- Speed is the GPS ground speed in m/s (`ground-speed`).

### From samples to bursts

A burst is what the model classifies: 20 acceleration samples 50 ms apart
(1 s at 20 Hz), with the ground speed of their GPS fix. A fix is one GPS
record. An acceleration sample belongs to the fix whose time is its
`start-timestamp`. The rules, in order:

1. **No missing values.** A sample without a time, fix time, tag, animal or
   acceleration value is dropped.
2. **No repeated samples.** A second sample with the same time in one fix is
   dropped.
3. **Consecutive samples.** Within one fix, a run goes on while each sample is
   50 ms after the one before. A gap, or another sampling rate, ends the run.
4. **Bursts of 20.** Each run is cut into bursts of 20 from its first sample.
   Leftover samples, and runs shorter than 20, are dropped. Example: a fix of
   50 samples with the 5th missing gives two bursts, samples 6-25 and 26-45;
   samples 1-4 and 46-50 are dropped.
5. **A matching GPS fix.** A burst takes the ground speed of its fix: same
   tag, same animal, fix time equal to `start-timestamp`. It is dropped if the
   fix is missing, has no speed, or has two different speeds.
6. **Speed below 30 m/s.** A faster burst is dropped: no gull flies that fast,
   so the speed is a GPS error.
7. **Acceleration clipped** to between -2 and 2 g, as in training.

The time of a burst is the time of its first sample. One fix can hold several
bursts, and each gets its own class.

## Model

The classifier is a small one-dimensional convolutional neural network (CNN)
with 5,429 trainable parameters. It reads one burst as 7 channels by 20 time
steps. Four channels are the data: the acceleration along the three axes of
the tag, in g, and the ground speed, repeated at every step. The speed is
divided by 22.3012351755624 m/s, the fastest speed in the labeled training
data; every file uses this same number, because the model learned this scale.
Three more channels are computed from the acceleration vector a(t) at each
step t. The first is its length |a(t)|, the total acceleration. The second is
|a(t) - m|, where m is the mean of the burst; this removes gravity and posture
and is the vectorial dynamic body acceleration (VeDBA). The third is
|a(t) - a(t-1)|, the change since the previous step (jerk); the first step
copies the second. Rotating the tag changes the three axes but not these three
lengths.

The network has three convolution layers. Each has 20 channels and a kernel of
5 steps, with a dilation of 1, 2 and 3, and is followed by group normalisation
(4 groups) and a GELU activation (Gaussian error linear unit). Padding keeps
the length at 20. Together the three layers see 25 steps, so each output step
depends on the whole burst. The output of the last layer is summarised over
time by the mean, maximum and standard deviation of each channel: 60 numbers.
After dropout (rate 0.15), a linear layer turns these into one score per class.
The class with the highest score is the prediction.

## Training

For reference: how the shipped weights were made. The labeled data below are
not the data the classifier runs on, and are not needed to use it.

### Labeled data

- 4,338 labeled bursts from 1,822 GPS fixes of 19 gulls, May 2012 to May 2016:
  - 14 lesser black-backed gulls (*Larus fuscus*) of the LBBG_TEXEL study,
    3,503 bursts, tags 533, 534, 537, 541, 606, 608, 754, 757, 781, 782, 798,
    805, 806 and 871;
  - 5 herring gulls (*Larus argentatus*) of the HG_TEXEL study, 835 bursts,
    tags 1600, 6011, 6016, 6080 and 6210.
- All tags are UvA-BiTS, with acceleration at 20 Hz.
- The data came from the UvA-BiTS database, not from the published files, and
  passed the same rules (Data, rules 1-7). There a sample was known by its
  place in the burst (an index) instead of its time: a run went on while the
  index went up by 1, and the fix was matched to the same second. Raw counts
  were converted to g with each tag's calibration (offset and sensitivity),
  which gives the same values as the published `tilt-x`, `tilt-y`, `tilt-z`.
- Experts labeled the data. Four labeled sets, made by different people, were
  merged. One is the set of Shamoun-Baranes et al. (2016), labeled from the
  acceleration signal and the GPS track (its Table S1). Where sets overlapped
  and disagreed, the label was corrected or removed.
- A run of samples with one label is cut into bursts of 20 from its first
  labeled sample, as in rule 4.
- The classes are those of the paper, in lower case:

| class | meaning (Table S1) | bursts |
|---|---|---|
| `flap` | flapping flight with regular wing beat | 643 |
| `exflap` | irregular, intense wing beat, at take-off or in aerial interactions | 38 |
| `soar` | flight with no wing beat, including climbing and gliding | 537 |
| `boat` | sitting or standing on a boat | 176 |
| `float` | floating with the currents at sea | 729 |
| `stationary` | sitting or standing on land, or on a static structure at sea | 1,502 |
| `walk` | walking | 337 |
| `mixed` | mixed flapping and gliding | 151 |
| `peck` | walking and pecking | 225 |

- The paper's class Other (a signal that fits no class) was removed.
- Two sets had extra classes. Float_groyne became `float` and Handling_mussel
  became `peck`; the others were removed.

### How it was trained

The 4,338 labeled bursts were split at random into a training set of 3,900
bursts (90%) and a validation set of 438 bursts (10%). The split was made per
class, so both sets have the same mix of classes. The model was trained for
4,000 epochs with all training bursts in one batch, so one update per epoch.
The loss was cross-entropy, without class weights. The optimiser was AdamW,
with a learning rate of 3 x 10^-4 and a weight decay of 10^-2; the learning
rate was divided by 10 after epoch 2,000.

In every epoch, each training burst was turned by a new random 3D rotation of
its three acceleration axes; the speed was not changed. How a tag sits on a
bird differs between birds and between tag types, so the model should not
depend on the orientation of the tag. Validation bursts were not rotated.

After each epoch, the accuracy on the validation set was measured, and the
weights of the best epoch, epoch 1,870, were kept. These weights ship with
this repository. Training took about 5 minutes on a laptop GPU (graphics
processing unit).

### Results

On the 438 validation bursts the model is right 94.5% of the time (97.9% on
the training set). Averaged over the 9 classes, each class counting equally,
the F1 score is 0.91 and the balanced accuracy is 89.8%. F1 is the harmonic
mean of precision and recall; balanced accuracy is the mean recall per class.

| class | validation bursts | F1 |
|---|---|---|
| `flap` | 65 | 0.99 |
| `exflap` | 4 | 1.00 |
| `soar` | 54 | 0.90 |
| `boat` | 18 | 0.94 |
| `float` | 73 | 0.97 |
| `stationary` | 151 | 0.98 |
| `walk` | 34 | 0.89 |
| `mixed` | 16 | 0.79 |
| `peck` | 23 | 0.76 |

Most errors are between classes that look alike in 1 s of data: 4 of 16
`mixed` bursts are called `soar`, 4 of 54 `soar` bursts are called `float`,
and of 23 `peck` bursts, 3 are called `walk`, 2 `soar` and 2 `stationary`.

## Limitations

- The split is by burst, not by bird or by time. One bird, and even
  neighbouring bursts of one fix, can be in both sets. The validation set also
  chose the epoch, and there is no separate test set. Expect lower accuracy on
  new birds than the figures above.
- Rare classes have few validation bursts (`exflap` 4, `mixed` 16, `boat` 18,
  `peck` 23), so their F1 is uncertain.
- The training data are lesser black-backed and herring gulls with UvA-BiTS
  tags at 20 Hz. Other species and tags were not tested. Bursts at other
  sampling rates are dropped by rule 3.
- Every burst gets one of the 9 classes, even when the bird does something
  else: the class Other was removed from training.
- 26 labeled bursts have a speed that contradicts their label, such as
  `stationary` at 3 m/s or more. They are in the training data of this model.
- The 19 labeled birds are in the published LBBG_TEXEL and HG_TEXEL data, so
  results on those birds may look better than on others.

## References

- Shamoun-Baranes J, Bouten W, van Loon EE, Meijer C, Camphuysen CJ (2016).
  Flap or soar? How a flight generalist responds to its aerial environment.
  Philosophical Transactions of the Royal Society B 371: 20150395.
  <https://doi.org/10.1098/rstb.2015.0395>
- Shamoun-Baranes J, van Donk S, Shinneman S, Govaert S, Desmet P,
  Camphuysen KCJ. HG_TEXEL - Herring gulls (*Larus argentatus*, Laridae)
  breeding in Texel (the Netherlands). Zenodo.
  <https://doi.org/10.5281/zenodo.20706394>
- Shamoun-Baranes J, Brown JM, Kentie R, Shinneman S, Govaert S, Desmet P,
  Camphuysen KCJ. LBBG_TEXEL - Lesser black-backed gulls (*Larus fuscus*,
  Laridae) breeding in Texel (the Netherlands). Zenodo.
  <https://doi.org/10.5281/zenodo.20706372>
