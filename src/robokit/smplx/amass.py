"""Build a `BodyModelState` from an AMASS-style SMPL-X NPZ dict."""

from typing import Any, Dict

import numpy as np

from robokit.smplx.smplx_constants import SMPLX_NUM_JOINTS
from robokit.smplx.spec import BodyModelSpec
from robokit.smplx.state import BodyModelState


def from_amass_dict(
    smplx_data: Dict[str, Any],
    spec: BodyModelSpec,
) -> BodyModelState:
    """Convert an AMASS-style SMPL-X NPZ dict into a `BodyModelState`.

    Accepts either `poses` (156 / 165 DoF, root + body + face + hands) or
    the split `root_orient` + `pose_body` form. Trims / zero-pads
    `betas` to `spec.num_betas`; face and hand DoFs are zeroed. Returns
    a numpy-backed state so the Warp runtime path never needs `torch`.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> import numpy as np
        >>> from robokit.smplx import load_smplx
        >>> spec = load_smplx("assets/body_models/smplx/SMPLX_NEUTRAL.npz")
        >>> data = {
        ...     "poses": np.zeros((4, 165), dtype=np.float32),
        ...     "trans": np.zeros((4, 3), dtype=np.float32),
        ...     "betas": np.zeros((10,), dtype=np.float32),
        ... }
        >>> state = from_amass_dict(data, spec)
        >>> state.full_pose_aa.shape
        (4, 55, 3)
        >>> state.betas.shape
        (4, 10)
        >>> state.transl.shape
        (4, 3)
    """
    if "pose_body" in smplx_data and "root_orient" in smplx_data:
        root_orient = np.asarray(smplx_data["root_orient"], dtype=np.float32)
        pose_body = np.asarray(smplx_data["pose_body"], dtype=np.float32)
    else:
        assert "poses" in smplx_data, "Expected either ('root_orient', 'pose_body') or 'poses' in smplx_data"
        poses = np.asarray(smplx_data["poses"], dtype=np.float32)
        root_orient = poses[:, :3]
        pose_body = poses[:, 3:66]
    assert root_orient.shape[1] == 3 and pose_body.shape[1] == 63, (
        f"Unexpected shapes: root_orient={root_orient.shape}, pose_body={pose_body.shape}"
    )
    num_frames = root_orient.shape[0]
    assert pose_body.shape[0] == num_frames

    betas_raw = np.asarray(smplx_data["betas"], dtype=np.float32).reshape(-1)
    nb = spec.num_betas
    if betas_raw.size >= nb:
        betas_trim = betas_raw[:nb]
    else:
        betas_trim = np.pad(betas_raw, (0, nb - betas_raw.size))
    betas_frames = np.ascontiguousarray(np.broadcast_to(betas_trim, (num_frames, nb)))

    transl = np.ascontiguousarray(np.asarray(smplx_data["trans"], dtype=np.float32))
    assert transl.shape == (num_frames, 3)

    full_pose_aa = np.zeros((num_frames, SMPLX_NUM_JOINTS, 3), dtype=np.float32)
    full_pose_aa[:, 0] = root_orient
    full_pose_aa[:, 1:22] = pose_body.reshape(num_frames, 21, 3)

    return BodyModelState(
        betas=betas_frames,
        full_pose_aa=full_pose_aa,
        transl=transl,
    )
