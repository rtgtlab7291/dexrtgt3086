"""Parametric body-model engine + SMPL-X and MANO format adapters.

Engine: LBS, FK, landmark gather. Reusable across SMPL / SMPL-H /
SMPL-X / MANO / FLAME. Format-specific loaders (`load_smplx`,
`load_mano`) build `BodyModelSpec` records and call into the
engine.

`body_lbs_torch` and `mano_forward_pca_mm` are exported lazily
(PEP 562) so that importing this package does not require `torch`.
"""

from typing import TYPE_CHECKING

from robokit.smplx.amass import from_amass_dict
from robokit.smplx.dexycb_state import from_dexycb_pose
from robokit.smplx.mano_constants import (
    MANO_FINGERTIP_NAMES,
    MANO_FINGERTIP_VERTEX_IDS_LEFT,
    MANO_FINGERTIP_VERTEX_IDS_RIGHT,
    MANO_JOINT_NAMES,
    MANO_NUM_JOINTS,
    MANO_NUM_PCA_COMPONENTS,
    MANO_NUM_POSE_DIRS,
    MANO_NUM_SHAPE_COEFFS,
    MANO_NUM_VERTICES,
    MANOPTH_KEYPOINT_PERMUTATION,
    Side,
    mano_fingertip_vertex_ids,
)
from robokit.smplx.mano_loader import load_mano
from robokit.smplx.smplx_constants import (
    SMPLX_JOINT_NAMES,
    SMPLX_NUM_JOINTS,
    SMPLX_NUM_POSE_DIRS,
    SMPLX_NUM_SHAPE_COEFFS,
    SMPLX_NUM_VERTICES,
    SMPLX_STATIC_LANDMARK_NAMES,
    SMPLX_STATIC_LANDMARK_VERTEX_IDS,
    Gender,
)
from robokit.smplx.smplx_loader import load_smplx
from robokit.smplx.spec import BodyModelSpec
from robokit.smplx.spec_tensors import BodyModelSpecTensors
from robokit.smplx.state import BodyModelState
from robokit.smplx.warp_lbs import body_fk_warp, body_lbs_warp


if TYPE_CHECKING:
    from robokit.smplx.mano_torch_forward import mano_forward_pca_mm
    from robokit.smplx.torch_lbs import body_lbs_torch


__all__ = [
    "BodyModelSpec",
    "BodyModelSpecTensors",
    "BodyModelState",
    "body_fk_warp",
    "body_lbs_torch",
    "body_lbs_warp",
    "from_amass_dict",
    "from_dexycb_pose",
    "Gender",
    "load_mano",
    "load_smplx",
    "mano_fingertip_vertex_ids",
    "mano_forward_pca_mm",
    "MANO_FINGERTIP_NAMES",
    "MANO_FINGERTIP_VERTEX_IDS_LEFT",
    "MANO_FINGERTIP_VERTEX_IDS_RIGHT",
    "MANO_JOINT_NAMES",
    "MANO_NUM_JOINTS",
    "MANO_NUM_PCA_COMPONENTS",
    "MANO_NUM_POSE_DIRS",
    "MANO_NUM_SHAPE_COEFFS",
    "MANO_NUM_VERTICES",
    "MANOPTH_KEYPOINT_PERMUTATION",
    "Side",
    "SMPLX_JOINT_NAMES",
    "SMPLX_NUM_JOINTS",
    "SMPLX_NUM_POSE_DIRS",
    "SMPLX_NUM_SHAPE_COEFFS",
    "SMPLX_NUM_VERTICES",
    "SMPLX_STATIC_LANDMARK_NAMES",
    "SMPLX_STATIC_LANDMARK_VERTEX_IDS",
]


def __getattr__(name: str):
    if name == "body_lbs_torch":
        from robokit.smplx.torch_lbs import body_lbs_torch

        return body_lbs_torch
    if name == "mano_forward_pca_mm":
        from robokit.smplx.mano_torch_forward import mano_forward_pca_mm

        return mano_forward_pca_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
