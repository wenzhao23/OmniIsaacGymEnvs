from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch


class ASampleHandView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "ASampleHandView",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self.actuated_joint_names = [
            "wrist_t_artificial_0", "artificial_0_t_1", "artificial_1_t_2",
            "artificial_2_t_3", "artificial_3_t_4", "artificial_4_t_5",
            "THJ1", "THJ2", "THJ3",
            "FFJ1", "FFJ2", "FFJ3",
            "MFJ2", "MFJ3",
            "RFJ2", "RFJ3",
            "LFJ2", "LFJ3",
        ]
        self._actuated_dof_indices = list()
        for joint_name in self.actuated_joint_names:
            self._actuated_dof_indices.append(self.get_dof_index(joint_name))
        self._actuated_dof_indices.sort()

        # limit_stiffness = torch.tensor([30.0] * self.num_fixed_tendons, device=self._device)
        # damping = torch.tensor([0.1] * self.num_fixed_tendons, device=self._device)
        # self.set_fixed_tendon_properties(dampings=damping, limit_stiffnesses=limit_stiffness)
