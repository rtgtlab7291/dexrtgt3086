"""Combo preset: Unitree G1 + Inspire hands, both sides.

The left binding sets the repair flags: the standalone left urdf uses a different fingertip
convention than the assembly, and collision spheres are authored for the right hand only.
"""

from robokit.assets.robots.hands import inspire_hand
from robokit.helpers.combo_retarget.config import ComboPreset, HandBinding
from robokit.helpers.humanoid_retarget.presets.g1_inspire_smplh import g1_inspire_smplh
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_RIGHT = HandBinding(
    urdf=str(inspire_hand.URDF_PATH),
    collision_spheres=str(inspire_hand.COLLISION_SPHERE_PATH),
    wrist_link="R_hand_base_link",
    joint_prefix="R_",
)

_LEFT = HandBinding(
    urdf=str(inspire_hand.URDF_PATH_LEFT),
    # authored for the right hand: the driver re-expresses every centre into left link frames
    collision_spheres=str(inspire_hand.COLLISION_SPHERE_PATH),
    wrist_link="L_hand_base_link",
    joint_prefix="L_",
    orientation_links=("L_thumb_proximal_base", "L_index_proximal", "L_pinky_proximal"),
    elbow_link="left_elbow_link",
    arm_prefix="left_",
    wrist_yaw_link="left_wrist_yaw_link",
    wrist_coord_spec=HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="-y"),
    mirror_spheres=True,
    align_tips_to_assembly=True,
)

g1_inspire_right = ComboPreset(body=g1_inspire_smplh, hands={"right": _RIGHT})
g1_inspire_bimanual = ComboPreset(body=g1_inspire_smplh, hands={"right": _RIGHT, "left": _LEFT})
