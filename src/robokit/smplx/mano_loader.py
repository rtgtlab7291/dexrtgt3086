"""Parse a MANO `.npz` model file into `BodyModelSpec`."""

from pathlib import Path
from typing import Union

import numpy as np

from robokit.smplx.mano_constants import (
    MANO_FINGERTIP_NAMES,
    MANO_JOINT_NAMES,
    MANO_NUM_JOINTS,
    MANO_NUM_POSE_DIRS,
    MANO_NUM_VERTICES,
    Side,
    mano_fingertip_vertex_ids,
)
from robokit.smplx.spec import BodyModelSpec


def load_mano(
    npz_path: Union[str, Path],
    side: Side = "right",
    num_betas: int = 10,
) -> BodyModelSpec:
    """Parse a MANO `.npz` and return a populated `BodyModelSpec`.

    `metadata` carries `side`, `hands_components` (45×45 PCA basis),
    and `hands_mean` (45-vec).

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> spec = load_mano("assets/body_models/mano/MANO_RIGHT.npz", side="right")
        >>> spec.v_template.shape
        (778, 3)
        >>> int(spec.parents[0])
        -1
        >>> spec.J_regressor.shape
        (16, 778)
        >>> spec.num_betas
        10
        >>> spec.metadata["side"]
        'right'
        >>> spec.static_landmark_vertex_ids.shape
        (5,)
    """
    data = np.load(str(npz_path), allow_pickle=False)

    v_template = data["v_template"].astype(np.float32)
    assert v_template.shape == (MANO_NUM_VERTICES, 3), (
        f"v_template {v_template.shape} != ({MANO_NUM_VERTICES}, 3); is this a MANO model?"
    )

    shapedirs_all = data["shapedirs"].astype(np.float32)
    assert shapedirs_all.shape[0] == MANO_NUM_VERTICES and shapedirs_all.shape[1] == 3
    assert num_betas <= shapedirs_all.shape[2], f"num_betas={num_betas} > shapedirs components={shapedirs_all.shape[2]}"
    shapedirs = np.ascontiguousarray(shapedirs_all[:, :, :num_betas])

    # NPZ stores posedirs as (V, 3, P); LBS consumes (P, V*3).
    posedirs_vxp = data["posedirs"].astype(np.float32)
    assert posedirs_vxp.shape == (MANO_NUM_VERTICES, 3, MANO_NUM_POSE_DIRS)
    posedirs = np.ascontiguousarray(posedirs_vxp.reshape(MANO_NUM_VERTICES * 3, MANO_NUM_POSE_DIRS).T)

    J_regressor = data["J_regressor"].astype(np.float32)
    assert J_regressor.shape == (MANO_NUM_JOINTS, MANO_NUM_VERTICES)

    # NPZ stores the root sentinel as MAX_UINT32-cast-to-int64; force to -1.
    parents = data["kintree_table"][0].astype(np.int64).copy()
    parents[0] = -1

    lbs_weights = data["weights"].astype(np.float32)
    assert lbs_weights.shape == (MANO_NUM_VERTICES, MANO_NUM_JOINTS)

    faces = data["f"].astype(np.int32)
    assert faces.ndim == 2 and faces.shape[1] == 3

    hands_components = data["hands_components"].astype(np.float32)
    hands_mean = data["hands_mean"].astype(np.float32)
    assert hands_components.shape == (45, 45)
    assert hands_mean.shape == (45,)

    recorded = str(data["side"])
    assert recorded == side, f"side mismatch: NPZ recorded {recorded!r} but caller passed {side!r}"

    return BodyModelSpec(
        name=f"mano_{side}",
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        lbs_weights=lbs_weights,
        faces=faces,
        joint_names=list(MANO_JOINT_NAMES),
        num_betas=num_betas,
        static_landmark_names=list(MANO_FINGERTIP_NAMES),
        static_landmark_vertex_ids=np.array(mano_fingertip_vertex_ids(side), dtype=np.int64),
        metadata={
            "side": side,
            "hands_components": hands_components,
            "hands_mean": hands_mean,
        },
    )
