from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})


from typing import Any, Mapping
import hydra
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omni.isaac.core.utils.stage import create_new_stage_async
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.maths import set_seed
# from omniisaacgymenvs.utils.hydra_cfg import reformat
from omniisaacgymenvs.robots.articulations import shadow_hand
from omniisaacgymenvs.utils.config_utils import sim_config
from omegaconf import DictConfig
import torch


class Scene:
    def __init__(self):
        self._hand_prim_path = "/World/kuka_allegro"
        self._stage = get_current_stage()
        self.device = "cuda:0"

    def get_hand(self):
        hand_start_translation = np.array([0.0, 0.0, 0.5])
        hand_start_orientation = np.array([0.0, 0.0, 0, 1])
        hand = shadow_hand.ShadowHand(
            prim_path=self._hand_prim_path,
            name="hand",
            translation=hand_start_translation, 
            orientation=hand_start_orientation,
        )
        hand.set_shadow_hand_properties(
            stage=self._stage, shadow_hand_prim=hand.prim)
        hand.set_motor_control_mode(
            stage=self._stage, shadow_hand_path=hand.prim_path)
        pose_dy, pose_dz = 0, 0.05
        return hand


def run():
    world_settings = {
        "physics_dt": 1.0 / 60.0,
        "stage_units_in_meters": 1.0,
        "rendering_dt": 1.0 / 60.0}
    world = World(**world_settings)
    world.scene.add_default_ground_plane()

    cube = world.scene.add(
        DynamicCuboid(
            name="cube",
            position=np.array([0.3, 0.3, 0.3]),
            prim_path="/World/Cube",
            scale=np.array([0.0515, 0.0515, 0.0515]),
            size=1.0,
            color=np.array([0, 0, 1]),
        )
    )

    scene = Scene(None)
    world.scene.add(scene.get_hand())
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == '__main__':
    run()
    simulation_app.close()
