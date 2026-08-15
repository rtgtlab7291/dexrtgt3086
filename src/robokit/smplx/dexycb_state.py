"""Build a `BodyModelState` from DexYCB-style MANO pose annotations."""

import numpy as np

from robokit.smplx.mano_constants import MANO_NUM_JOINTS, MANO_NUM_PCA_COMPONENTS
from robokit.smplx.spec import BodyModelSpec
from robokit.smplx.state import BodyModelState


def from_dexycb_pose(
    pose_m: np.ndarray,
    betas: np.ndarray,
    spec: BodyModelSpec,
    flat_hand_mean: bool = False,
) -> BodyModelState:
    """Convert DexYCB `pose_m` (`[B, 1, 51]` or `[B, 51]`) into a state.

    Layout: `[0:3]` global orient, `[3:48]` 45-D PCA coefficients,
    `[48:51]` translation. Adds `spec.metadata["hands_mean"]` to the
    PCA-expanded axis-angle when `flat_hand_mean=False` (DexYCB default).
    Returns a numpy-backed state - the Warp runtime path does not need
    `torch`.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> import numpy as np
        >>> from robokit.smplx import load_mano
        >>> spec = load_mano("assets/body_models/mano/MANO_RIGHT.npz", side="right")
        >>> pose_m = np.zeros((4, 1, 51), dtype=np.float32)
        >>> betas = np.zeros((10,), dtype=np.float32)
        >>> state = from_dexycb_pose(pose_m, betas, spec, flat_hand_mean=True)
        >>> state.full_pose_aa.shape
        (4, 16, 3)
        >>> state.betas.shape
        (4, 10)
    """
    pose_m = np.asarray(pose_m, dtype=np.float32)
    if pose_m.ndim == 3:
        assert pose_m.shape[1] == 1, f"unexpected pose_m shape {pose_m.shape}"
        pose_m = pose_m[:, 0, :]
    assert pose_m.ndim == 2 and pose_m.shape[1] == 51, f"pose_m must be [B, 51] (or [B, 1, 51]); got {pose_m.shape}"
    B = pose_m.shape[0]

    global_orient = pose_m[:, :3]
    pca_coeffs = pose_m[:, 3 : 3 + MANO_NUM_PCA_COMPONENTS]
    transl = np.ascontiguousarray(pose_m[:, 3 + MANO_NUM_PCA_COMPONENTS :])

    hands_components = spec.metadata["hands_components"]
    hands_mean = spec.metadata["hands_mean"]
    finger_aa = pca_coeffs @ hands_components
    if not flat_hand_mean:
        finger_aa = finger_aa + hands_mean

    full_pose_aa = np.zeros((B, MANO_NUM_JOINTS, 3), dtype=np.float32)
    full_pose_aa[:, 0] = global_orient
    full_pose_aa[:, 1:] = finger_aa.reshape(B, MANO_NUM_JOINTS - 1, 3)

    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = np.ascontiguousarray(np.broadcast_to(betas, (B, betas.shape[0])))
    assert betas.shape == (B, spec.num_betas), f"betas shape {betas.shape} != ({B}, {spec.num_betas})"

    return BodyModelState(
        betas=betas,
        full_pose_aa=full_pose_aa,
        transl=transl,
    )
