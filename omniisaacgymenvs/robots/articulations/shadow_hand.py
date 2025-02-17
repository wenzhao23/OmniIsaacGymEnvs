# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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


class ShadowHand(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "shadow_hand",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        assets_root_path = get_assets_root_path()
        assets_user_path = assets_root_path + "/../../../../my_data"
        self._usd_path = assets_user_path + "/allegro.usd"

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

    def set_shadow_hand_properties(self, stage, shadow_hand_prim):
        for link_prim in shadow_hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, shadow_hand_path):
        joints_config = {
                         "wrist_prismatic_1": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/allegro_mount/allegro_mount_0_1"},
                         "wrist_prismatic_2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/allegro_mount_1/allegro_mount_1_2"},
                         "wrist_prismatic_3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/allegro_mount_2/allegro_mount_2_3"},
                         "wrist_revoluate_1": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/allegro_mount_3/allegro_mount_3_4"},
                         "wrist_revoluate_2": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/allegro_mount_4/allegro_mount_4_5"},
                         "wrist_revoluate_3": {"stiffness": 10000, "damping": 0.1, "max_force": 10000,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/allegro_mount_5/allegro_mount_5_6"},
                         "index_joint_0": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/palm_link/index_joint_0"},
                         "index_joint_1": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/index_link_0/index_joint_1"},
                         "index_joint_2": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/index_link_1/index_joint_2"},
                         "index_joint_3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/index_link_2/index_joint_3"},
                         "middle_joint_0": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/palm_link/middle_joint_0"},
                         "middle_joint_1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/middle_link_0/middle_joint_1"},
                         "middle_joint_2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/middle_link_1/middle_joint_2"},
                         "middle_joint_3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/middle_link_2/middle_joint_3"},
                         "ring_joint_0": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/palm_link/ring_joint_0"},
                         "ring_joint_1": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/ring_link_0/ring_joint_1"},
                         "ring_joint_2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/ring_link_1/ring_joint_2"},
                         "ring_joint_3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/ring_link_2/ring_joint_3"},
                         "thumb_joint_0": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/palm_link/thumb_joint_0"},
                         "thumb_joint_1": {"stiffness": 1, "damping": 0.1, "max_force": 2.3722,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/thumb_link_0/thumb_joint_1"},
                         "thumb_joint_2": {"stiffness": 1, "damping": 0.1, "max_force": 1.45,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/thumb_link_1/thumb_joint_2"},
                         "thumb_joint_3": {"stiffness": 1, "damping": 0.1, "max_force": 0.99,
                                           "prim_path": shadow_hand_path + "/kuka_allegro/thumb_link_2/thumb_joint_3"},
                        }

        joint_names = [
            "wrist_prismatic_1", "wrist_prismatic_2", "wrist_prismatic_3",
            "wrist_revoluate_1", "wrist_revoluate_2", "wrist_revoluate_3",
            "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
            "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
            "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
            "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"
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
