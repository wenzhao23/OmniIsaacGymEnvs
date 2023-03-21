from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import pickle
import time
from typing import Any, Mapping

import numpy as np
from omni.isaac.core import World
from omniisaacgymenvs.tasks import grasping


def run():
  np.random.seed(17)

  # Creates the world  
  world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0}
  world = World(**world_settings)
  grasping_task = grasping.Grasping(
    "grasping_task_001", offset=np.array([0, 0, 0]))
  grasping_task.setup_scene(world.scene)
  world.reset()



if __name__ == '__main__':
  run()
  simulation_app.close()
