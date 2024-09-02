from tiny_tamp.inverse_kinematics.ikfast import *  # For legacy purposes
from tiny_tamp.inverse_kinematics.utils import IKFastInfo

FRANKA_URDF = "models/franka_description/robots/panda_arm_hand.urdf"

PANDA_INFO = IKFastInfo(
    module_name="franka_panda.ikfast_panda_arm",
    base_link="panda_link0",
    ee_link="panda_link8",
    free_joints=["panda_joint7"],
)
