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

  # TODO: use cloner for duplication
  tasks = []
  task_index = 0
  rows = 1
  cols = 1
  spacing = 0.5
  for i in range (rows):
    for j in range(cols):
      task_index += 1
      tasks.append(grasping.Grasping(
        f"grasping_{task_index:03d}",
        world,
        offset=spacing * np.array([i - int(rows / 2), j - int(cols / 2) , 0])))
      tasks[-1].setup_scene(world.scene)
  world.reset()

  # for task in tasks:
  #   for hand_view in task._hand_views.values():
  #     current_poses = hand_view.get_world_poses()
  #     hand_view.set_world_poses(positions=current_poses[0] + task.offset.reshape((1, 3)),
  #                             orientations=current_poses[1])

  time_elapsed = 0
  start_time = time.time()
  while simulation_app.is_running() and time_elapsed < 30:
    world.step(render=True)
    time_elapsed = time.time() - start_time


if __name__ == '__main__':
  run()
  simulation_app.close()
