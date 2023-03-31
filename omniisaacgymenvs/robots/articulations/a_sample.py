from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

import carb
from pxr import Usd, UsdGeom, Sdf, Gf, PhysxSchema, UsdPhysics


class ASample(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "a_sample_hand",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        assets_root_path = get_assets_root_path()
        assets_user_path = assets_root_path + "/../../../../my_data"
        self._usd_path = assets_user_path + "/hand/a_sample_v3.usd"

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(self._usd_path)

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_shadow_hand_properties(self, stage, hand_prim):
        hand_prim = get_prim_at_path("/World/aSampleForearm/aSampleForearm")
        for link_prim in hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, hand_path):
        joints_config = {
            "wrist_prismatic_1": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                  "prim_path": hand_path + "/Wrist_MASTER_Asm__1/wrist_t_artificial_0"},
            "wrist_prismatic_2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                  "prim_path": hand_path + "/artificial_0/artificial_0_t_1"},
            "wrist_prismatic_3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                  "prim_path": hand_path + "/artificial_1/artificial_1_t_2"},
            "wrist_revolute_1": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                  "prim_path": hand_path + "/artificial_2/artificial_2_t_3"},
            "wrist_revolute_2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                  "prim_path": hand_path + "/artificial_3/artificial_3_t_4"},
            "wrist_revolute_3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                  "prim_path": hand_path + "/artificial_4/artificial_4_t_5"},
            "thumb_j1": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Palm_Kinematic_Asm__1/THJ1"},
            "index_j1": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Palm_Kinematic_Asm__1/FFJ1"},
            "middle_j2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Palm_Kinematic_Asm__1/MFJ2"},
            "ring_j2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Palm_Kinematic_Asm__1/RFJ2"},
            "little_j2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Palm_Kinematic_Asm__1/LFJ2"},
            "index_j2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Knuckle_Asm_2__1/FFJ2"},
            "index_j3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Proximal_Asm__1/FFJ3"},
            "middle_j3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Proximal_Asm__2/MFJ3"},
            "ring_j3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Proximal_Asm__3/RFJ3"},
            "little_j3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Proximal_Asm__4/LFJ3"},
            "thumb_j2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Thumb_Knuckle_2___Asm__1/THJ2"},
            "thumb_j3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                         "prim_path": hand_path + "/Thumb_Proximal_2___Asm__1/THJ3"},
        }


        joint_names = [
            "wrist_prismatic_1", "wrist_prismatic_2", "wrist_prismatic_3",
            "wrist_revolute_1", "wrist_revolute_2", "wrist_revolute_3",
            "thumb_j1", "thumb_j2", "thumb_j3",
            "index_j1", "index_j2", "index_j3",
            "middle_j2", "middle_j3",
            "ring_j2", "ring_j3",
            "little_j2", "little_j3", 
        ]

        for joint_name in joint_names:
            config = joints_config[joint_name]
            if "prismatic" in joint_name:
                drive_type = "linear"
                stiffness = config["stiffness"]
                damping = config["damping"]
            else:
                drive_type = "angular"
                stiffness = config["stiffness"]*np.pi/180
                damping = config["damping"]*np.pi/180
            set_drive(
                # f"{self.prim_path}/joints/{joint_name}", 
                config["prim_path"],
                drive_type, 
                "position", 
                0.0, 
                stiffness,
                damping,
                config["max_force"]
            )
