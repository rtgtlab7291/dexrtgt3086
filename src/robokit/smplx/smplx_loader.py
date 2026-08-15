"""Parse a SMPL-X `.npz` model file into `BodyModelSpec`."""

from pathlib import Path
from typing import Union

import numpy as np

from robokit.smplx.smplx_constants import (
    SMPLX_JOINT_NAMES,
    SMPLX_NUM_JOINTS,
    SMPLX_NUM_POSE_DIRS,
    SMPLX_NUM_VERTICES,
    SMPLX_STATIC_LANDMARK_NAMES,
    SMPLX_STATIC_LANDMARK_VERTEX_IDS,
    Gender,
)
from robokit.smplx.spec import BodyModelSpec


def load_smplx(
    npz_path: Union[str, Path],
    gender: Gender = "neutral",
    num_betas: int = 10,
) -> BodyModelSpec:
    """Parse a SMPL-X `.npz` and return a populated `BodyModelSpec`.

    `metadata` carries `gender`, `hands_meanl`, and `hands_meanr`.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> spec = load_smplx("assets/body_models/smplx/SMPLX_NEUTRAL.npz")
        >>> spec.v_template.shape
        (10475, 3)
        >>> int(spec.parents[0])
        -1
        >>> spec.J_regressor.shape
        (55, 10475)
        >>> spec.num_betas
        10
        >>> spec.metadata["gender"]
        'neutral'
    """
    data = np.load(str(npz_path), allow_pickle=False)

    v_template = data["v_template"].astype(np.float32)
    assert v_template.shape == (SMPLX_NUM_VERTICES, 3), (
        f"v_template {v_template.shape} != ({SMPLX_NUM_VERTICES}, 3); is this a SMPL-X model?"
    )

    shapedirs_all = data["shapedirs"].astype(np.float32)
    assert shapedirs_all.shape[0] == SMPLX_NUM_VERTICES and shapedirs_all.shape[1] == 3
    assert num_betas <= shapedirs_all.shape[2], f"num_betas={num_betas} > shapedirs components={shapedirs_all.shape[2]}"
    shapedirs = np.ascontiguousarray(shapedirs_all[:, :, :num_betas])

    # NPZ stores posedirs as (V, 3, P); LBS consumes (P, V*3).
    posedirs_vxp = data["posedirs"].astype(np.float32)
    assert posedirs_vxp.shape == (SMPLX_NUM_VERTICES, 3, SMPLX_NUM_POSE_DIRS)
    posedirs = np.ascontiguousarray(posedirs_vxp.reshape(SMPLX_NUM_VERTICES * 3, SMPLX_NUM_POSE_DIRS).T)

    J_regressor = data["J_regressor"].astype(np.float32)
    assert J_regressor.shape == (SMPLX_NUM_JOINTS, SMPLX_NUM_VERTICES)

    # NPZ stores the root sentinel as MAX_UINT32-cast-to-int64; force to -1.
    parents = data["kintree_table"][0].astype(np.int64).copy()
    parents[0] = -1

    lbs_weights = data["weights"].astype(np.float32)
    assert lbs_weights.shape == (SMPLX_NUM_VERTICES, SMPLX_NUM_JOINTS)

    faces = data["f"].astype(np.int32)
    assert faces.ndim == 2 and faces.shape[1] == 3

    hands_meanl = data["hands_meanl"].astype(np.float32)
    hands_meanr = data["hands_meanr"].astype(np.float32)
    assert hands_meanl.shape == (45,) and hands_meanr.shape == (45,)

    return BodyModelSpec(
        name=f"smplx_{gender}",
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        lbs_weights=lbs_weights,
        faces=faces,
        joint_names=list(SMPLX_JOINT_NAMES),
        num_betas=num_betas,
        static_landmark_names=list(SMPLX_STATIC_LANDMARK_NAMES),
        static_landmark_vertex_ids=np.array(SMPLX_STATIC_LANDMARK_VERTEX_IDS, dtype=np.int64),
        metadata={
            "gender": gender,
            "hands_meanl": hands_meanl,
            "hands_meanr": hands_meanr,
        },
    )
