import numpy as np
import pytest

from gulliver import model as model_module
from gulliver.model import CLASS_NAMES, GPS_SPEED_SCALE, classify, model_input


def test_shipped_weights_fit_the_network(classifier):
    assert classifier.net[0].in_channels == 7
    assert classifier.fc.out_features == len(CLASS_NAMES)


def test_model_input_has_seven_channels():
    still = np.zeros((1, 20, 3), dtype=np.float32)
    still[:, :, 2] = 1.0  # 1 g on z, the bird is not moving

    data = model_input(still, np.array([GPS_SPEED_SCALE], dtype=np.float32))

    assert data.shape == (1, 7, 20)
    assert np.allclose(data[0, 3], 1.0)  # speed, divided by the training scale
    assert np.allclose(data[0, 4], 1.0)  # mag, gravity only
    assert np.allclose(data[0, 5], 0.0)  # dyn_mag, no motion
    assert np.allclose(data[0, 6], 0.0)  # jerk_mag, nothing changes


def test_classify_returns_known_names(classifier):
    bursts = np.random.default_rng(0).random((4, 20, 3)).astype(np.float32)
    speeds = np.full(4, 2.0, dtype=np.float32)

    names, confidence = classify(bursts, speeds, classifier)

    assert set(names) <= set(CLASS_NAMES)
    assert ((confidence > 0) & (confidence <= 1)).all()


def test_batching_does_not_change_the_result(classifier, monkeypatch):
    bursts = np.random.default_rng(1).random((5, 20, 3)).astype(np.float32)
    speeds = np.full(5, 2.0, dtype=np.float32)
    whole, _ = classify(bursts, speeds, classifier)

    monkeypatch.setattr(model_module, "BATCH_SIZE", 2)
    in_pieces, _ = classify(bursts, speeds, classifier)

    assert whole.tolist() == in_pieces.tolist()


def test_empty_input_gives_empty_output(classifier):
    names, confidence = classify(
        np.empty((0, 20, 3), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        classifier,
    )

    assert len(names) == 0 and len(confidence) == 0


@pytest.mark.parametrize("bad_channel", [0, 1, 2])
def test_one_axis_at_a_time_is_still_classified(classifier, bad_channel):
    burst = np.zeros((1, 20, 3), dtype=np.float32)
    burst[0, :, bad_channel] = 1.0

    names, _ = classify(burst, np.array([1.0], dtype=np.float32), classifier)

    assert names[0] in CLASS_NAMES
