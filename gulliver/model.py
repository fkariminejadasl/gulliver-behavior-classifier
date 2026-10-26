"""The trained behaviour classifier.

The network is `BirdModelSmallDilated`, with the trained weights in
`models/197_best.pth`. It reads one burst as 7 channels by 20 samples: the
three acceleration axes, the scaled GPS speed, and three magnitude channels. The weights only load into this exact shape, so a change
to the network fails at loading instead of going unnoticed.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

CHECKPOINT = Path(__file__).parent / "models" / "197_best.pth"
BATCH_SIZE = 4096

# The 9 classes the model was trained on, in its output order. The class Other
# was left out of training. The names are those of Shamoun-Baranes et al. (2016),
# lowercased. In training, stationary was sitstand, walk terloco, mixed manouvre,
# peck pecking.
CLASS_NAMES = [
    "flap",
    "exflap",
    "soar",
    "boat",
    "float",
    "stationary",
    "walk",
    "mixed",
    "peck",
]
# The speed the training data was divided by: the fastest speed in the labeled
# training data, in m/s.
GPS_SPEED_SCALE = 22.3012351755624


class BirdModelSmallDilated(nn.Module):
    def __init__(self, in_channels=7, mid_channels=20, out_channels=9, dropout=0.15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(4, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(4, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=6, dilation=3),
            nn.GroupNorm(4, mid_channels),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(mid_channels * 3, out_channels)

    def forward(self, x):
        x = self.net(x)
        x = torch.cat(
            [
                x.mean(dim=-1),
                x.amax(dim=-1),
                x.std(dim=-1, unbiased=False),
            ],
            dim=1,
        )
        return self.fc(self.dropout(x))


def add_magnitude_features(data: np.ndarray) -> np.ndarray:
    """N x L x 4 (acceleration x, y, z, GPS speed) -> N x L x 7.

    The three added channels are norms of an acceleration vector, so a rotation of
    the whole burst leaves them alone:

      mag      total acceleration whatever its direction, about 1 g when the
               bird is still, because only gravity is left.
      dyn_mag  what is left after the static part (gravity and posture) is
               removed, that is VeDBA (Vectorial Dynamic Body Acceleration, the
               usual movement-intensity measure in bio-logging) taken over the
               mean of the burst.
      jerk_mag how abruptly the acceleration changes, per sample. It is
               undefined at the first sample and repeats the second one there.
    """
    acc = data[:, :, :3]
    mag = np.linalg.norm(acc, axis=2)
    dyn_mag = np.linalg.norm(acc - acc.mean(axis=1, keepdims=True), axis=2)
    jerk_mag = np.linalg.norm(np.diff(acc, axis=1), axis=2)  # L-1 long
    jerk_mag = np.concatenate([jerk_mag[:, :1], jerk_mag], axis=1)  # edge-replicate
    return np.concatenate([data, np.stack([mag, dyn_mag, jerk_mag], axis=2)], axis=2)


def load_classifier(checkpoint: str | Path = CHECKPOINT) -> nn.Module:
    """Build the network and load the trained weights into it."""
    model = BirdModelSmallDilated()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def model_input(acceleration: np.ndarray, ground_speed: np.ndarray) -> np.ndarray:
    """N x 20 x 3 bursts and N speeds -> N x 7 x 20, what the network reads.

    The steps and their order are those used in training.
    """
    speed = np.repeat(
        ground_speed[:, None, None], acceleration.shape[1], axis=1
    )  # N x L x 1
    data = np.concatenate([acceleration, speed], axis=2)  # N x L x 4
    data = add_magnitude_features(data)  # N x L x 7
    data[:, :, 3] /= GPS_SPEED_SCALE
    return np.ascontiguousarray(data.transpose(0, 2, 1), dtype=np.float32)


def classify(
    acceleration: np.ndarray, ground_speed: np.ndarray, model: nn.Module | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one class name and its probability per burst."""
    if model is None:
        model = load_classifier()

    names = np.empty(len(acceleration), dtype=object)
    confidence = np.empty(len(acceleration), dtype=np.float32)
    for start in range(0, len(acceleration), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = model_input(acceleration[start:end], ground_speed[start:end])
        with torch.no_grad():
            logits = model(torch.from_numpy(batch))
        probabilities = torch.softmax(logits, dim=-1)
        best = torch.argmax(logits, dim=1)
        names[start:end] = [CLASS_NAMES[index] for index in best.tolist()]
        confidence[start:end] = probabilities.gather(1, best[:, None])[:, 0].numpy()
    return names, confidence
