import numpy as np
import pytest

from robokit.utils.hand_coord_utils import HandCoordinateSpec, hand_coord_conversion


# legacy coord strings used across consumers, with their expected (forward, up) axes
LEGACY_TO_AXES = {
    "x: front, y: left, z: up": ("x", "z"),
    "x: left, y: front, z: down": ("y", "-z"),
    "x: back, y: down, z: right": ("-x", "-y"),
    "x: left, y: back, z: up": ("-y", "z"),
    "x: down, y: left, z: front": ("z", "-x"),
    "x: right, y: front, z: up": ("y", "z"),
}


def test_from_legacy_string_parses_forward_and_up():
    for legacy, (forward, up) in LEGACY_TO_AXES.items():
        spec = HandCoordinateSpec.from_legacy_string(legacy)
        assert spec.palm_forward_axis == forward, legacy
        assert spec.four_fingers_up_axis == up, legacy


def test_from_legacy_string_accepts_preset_names():
    from robokit.xform.utils import COMMON_COORDS

    for preset, coord in COMMON_COORDS.items():
        assert HandCoordinateSpec.from_legacy_string(preset) == HandCoordinateSpec.from_legacy_string(coord), preset


def test_hand_coord_conversion_return_tensors_pt_matches_np():
    pytest.importorskip("torch")
    specs = [HandCoordinateSpec.from_legacy_string(s) for s in LEGACY_TO_AXES]
    for a in specs:
        for b in specs:
            np_mat = hand_coord_conversion(a, b)
            pt_mat = hand_coord_conversion(a, b, return_tensors="pt")
            assert np.allclose(np_mat, pt_mat.numpy(), atol=1e-6)
