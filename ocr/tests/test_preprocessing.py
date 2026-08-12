import numpy as np
import pytest
from ocr_extraction.preprocessing import (
    ORIENTATION_ANGLES,
    estimate_orientation,
    rotate_counterclockwise,
)


@pytest.mark.parametrize("angle", ORIENTATION_ANGLES)
def test_manual_orientation_supports_every_right_angle(angle: int) -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    detected, score, method, candidates = estimate_orientation(image, str(angle))

    assert detected == angle
    assert score == 1.0
    assert method == "manual"
    assert candidates == (angle,)


def test_quarter_turn_swaps_dimensions() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    assert rotate_counterclockwise(image, 90).shape[:2] == (12, 8)
    assert rotate_counterclockwise(image, 270).shape[:2] == (12, 8)
