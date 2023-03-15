from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import time
from typing import Any, Mapping

from data_types import se3
import hydra
import kinpy as kp
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
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.sensor import ContactSensor
# from omniisaacgymenvs.utils.hydra_cfg import reformat
from omniisaacgymenvs.robots.articulations import shadow_hand
from omniisaacgymenvs.robots.articulations.views import shadow_hand_view
from omniisaacgymenvs.utils.config_utils import sim_config
from omegaconf import DictConfig
import torch


class Scene:
  def __init__(self):
    self._hand_prim_path = "/World/kuka_allegro"
    self._stage = get_current_stage()
    self.device = "cuda:0"

  def get_hand(self):
    hand_start_translation = np.array([-0.1, 0.0, 0.6])
    hand_start_orientation = (
      se3.Transform(rot=np.radians([0, 120, 0])) *
      se3.Transform(rot=np.radians([0, 0, 140]))
    ).quaternion
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

    hand_view = shadow_hand_view.ShadowHandView(
      prim_paths_expr="/World/kuka_allegro/kuka_allegro",
      name="shadow_hand_view")
    return hand, hand_view


def add_objects(world):
  physics_material = PhysicsMaterial("/World/PhysicsMaterial")
  num_random = 0
  min_xyz = [-0.2, -0.2, 0.15]
  max_xyz = [0.2, 0.2, 0.4]
  for i in range(num_random):
    position = np.random.uniform(low=min_xyz, high=max_xyz)
    world.scene.add(
      DynamicCuboid(
        name=f"cube{i}",
        position=position, # np.array([0, 0, 0.3]),
        prim_path=f"/World/Cube{i}",
        scale=np.array([0.0515, 0.0515, 0.0515]),
        size=1.0,
        color=np.array([0, 0, 1]),
        physics_material=physics_material
      )
    )
  world.scene.add(
    DynamicCuboid(
      name=f"cube_support",
      position=np.array([0, 0, 0.3]),
      prim_path=f"/World/CubeSupport",
      scale=np.array([0.0515, 0.0515, 0.1015]),
      size=1.0,
      color=np.array([0, 0, 1]),
      physics_material=physics_material
    )
  )
  world.scene.add(
    DynamicCuboid(
      name=f"cube_lift",
      position=np.array([0, 0, 0.4]),
      prim_path=f"/World/CubeLift",
      scale=np.array([0.0515, 0.1515, 0.0515]),
      size=1.0,
      color=np.array([0, 0, 1]),
      physics_material=physics_material
    )
  )


def run():
  np.random.seed(17)
  world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0}
  world = World(**world_settings)
  world.scene.add_default_ground_plane()

  add_objects(world)

  scene = Scene()
  hand, hand_view = scene.get_hand()
  world.scene.add(hand_view)
  chain = kp.build_chain_from_urdf(
    open(os.path.expanduser('~/Downloads/allegro/allegro.urdf')).read())

  for prim in scene._stage.TraverseAll():
    prim_type = prim.GetTypeName()
    if "Joint" in prim_type:
      print(prim)

  hand_base_prim_path = "/World/kuka_allegro/kuka_allegro/"
  contact_links = [
    "index_link_2", "index_link_3",
    "thumb_link_2", "thumb_link_3"]
    # "index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip",
    # "index_link_1", "index_link_2", "index_link_3",
    # "middle_link_1", "middle_link_2", "middle_link_3",
    # "ring_link_1", "ring_link_2", "ring_link_3",
    # "thumb_link_1", "thumb_link_2", "thumb_link_3"]
  hand_sensors = []
  for contact_link in contact_links: 
    hand_sensors.append(world.scene.add(
      ContactSensor(
        prim_path=hand_base_prim_path + contact_link + "/contact_sensor",
        name="contact_sensor_" + contact_link,
        min_threshold=0,
        max_threshold=10000000,
        radius=0.01,
      )
    ))
    hand_sensors[-1].add_raw_contact_data_to_frame()

  world.reset()

  # TODO: organize to state machine

  start_time = time.time()
  world_t_base_vec = hand_view.get_world_poses([0])
  world_t_initial = se3.Transform(
    xyz=world_t_base_vec[0][0], rot=world_t_base_vec[1][0])
  thumb_joint_0_index = hand_view.dof_names.index("thumb_joint_0")
  wrist_names = ["allegro_mount_0_1",
                 "allegro_mount_1_2",
                 "allegro_mount_2_3",
                 "allegro_mount_3_4",
                 "allegro_mount_4_5",
                 "allegro_mount_5_6"]
  wrist_indices = [hand_view.dof_names.index(name) for name in wrist_names]
  finger_names = ["thumb_joint_0",
                  "thumb_joint_2",
                  "thumb_joint_3",
                  "index_joint_1",
                  "index_joint_2",
                  "index_joint_3",
                  "middle_joint_1",
                  "middle_joint_2",
                  "middle_joint_3",
                  "ring_joint_1",
                  "ring_joint_2",
                  "ring_joint_3"]

  finger_indices = [hand_view.dof_names.index(name) for name in finger_names]
  world_t_grasp = se3.Transform(xyz=[0, 0, -0.4]) * world_t_initial
  initial_t_grasp = world_t_initial.inverse() * world_t_grasp
  grasp_duration = 5.0
  retract_duration = 3.0
  close_time = 5.0
  num_contacts = 0
  in_contact = False
  grasped = False
  retracting = False
  # print(wrist_indices)
  # print(finger_indices)
  time_elapsed = 0
  # import sys
  # sys.exit(1)
  while simulation_app.is_running() and time_elapsed < 10:
    world.step(render=True)
    command = [0] * 22
    if grasped:
      if not retracting:
        world_t_retract = se3.Transform(xyz=[0, 0, 0.4]) * world_t_actual_grasp
        actual_grasp_t_retract = world_t_actual_grasp.inverse() * world_t_retract
        actual_grasp_t_retract = initial_t_grasp.inverse()
        time_retract = time_elapsed
        finger_joint_grasp = hand_view.get_joint_positions()[0][finger_indices]
        retracting = True
      command[:3] = actual_grasp_joint_vec[:3] + actual_grasp_t_retract.translation * min(
        1, (time_elapsed - time_retract) / retract_duration)
      for i, index in enumerate(finger_indices):
        command[index] = finger_joint_grasp[i]
    else:
      if not in_contact:
        command[:3] = initial_t_grasp.translation * min(1, time_elapsed / grasp_duration)
      else:
        command[:3] = actual_grasp_joint_vec[:3]
      if time_elapsed > close_time:
        for i, index in enumerate(finger_indices):
          command[index] = 0.3 * (time_elapsed - close_time)

    command[thumb_joint_0_index] = np.radians(90.0)
    hand_view.set_joint_position_targets(
      command
    )
    # print('-' * 50)
    # print(np.linalg.norm(
    #   hand_view._physics_view.get_force_sensor_forces()[0, :, :], axis=1))

    print('-' * 30)

    joint_config = {}
    wrist_joint_vec = hand_view.get_joint_positions()[0][wrist_indices]
    for joint_name, joint_pos in zip(wrist_names, wrist_joint_vec):
      joint_config[joint_name] = joint_pos
    finger_joint_vec = hand_view.get_joint_positions()[0][finger_indices]
    for joint_name, joint_pos in zip(finger_names, finger_joint_vec):
      joint_config[joint_name] = joint_pos
    fk_result = chain.forward_kinematics(joint_config)
    print(fk_result["allegro_mount_6"])
    for hs in hand_sensors:
      print(hs.get_current_frame())

    num_contacts = [hand_sensor.get_current_frame()["number_of_contacts"]
            for hand_sensor in hand_sensors]
    if in_contact:
      if time_elapsed - time_in_contact > 5.0:
        grasped = True
    else:
      if np.mean(num_contacts) > 3:
        in_contact = True
        print("-" * 10 + "in_contact")
        time_in_contact = time_elapsed
        actual_grasp_joint_vec = hand_view.get_joint_positions()[0][wrist_indices]
        initial_t_actual_grasp = se3.Transform(
          xyz=actual_grasp_joint_vec[:3], rot=actual_grasp_joint_vec[3:])
        world_t_actual_grasp = world_t_initial * initial_t_actual_grasp
    time_elapsed = time.time() - start_time


if __name__ == '__main__':
  run()
  simulation_app.close()
