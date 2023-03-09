from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import time
from typing import Any, Mapping

import cv2
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
from omniisaacgymenvs.minimal_hand import config
from omniisaacgymenvs.minimal_hand.hand_mesh import HandMesh
from omniisaacgymenvs.minimal_hand.capture import OpenCVCapture
from omniisaacgymenvs.minimal_hand.kinematics import mpii_to_mano
from omniisaacgymenvs.minimal_hand.utils import OneEuroFilter, imresize
from omniisaacgymenvs.minimal_hand.wrappers import ModelPipeline
from omniisaacgymenvs.minimal_hand.utils import *

from omniisaacgymenvs.data_types import se3
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
    hand_start_translation = np.array([0.0, 0.0, 0.1])
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


def run(capture):

  ############ input and output visualization ############
  pygame.init()
  window_size = 1080
  display = pygame.display.set_mode((window_size, window_size))
  pygame.display.set_caption('Minimal Hand - input')

  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()
  model = ModelPipeline()

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

  for prim in scene._stage.TraverseAll():
    prim_type = prim.GetTypeName()
    if "Joint" in prim_type:
      print(prim)

  hand_base_prim_path = "/World/kuka_allegro/kuka_allegro/"
  contact_links = ["index_link_1", "index_link_2", "index_link_3",
           "middle_link_1", "middle_link_2", "middle_link_3",
           "ring_link_1", "ring_link_2", "ring_link_3",
           "thumb_link_1", "thumb_link_2", "thumb_link_3"]
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
  wrist_indices = [
    hand_view.dof_names.index("allegro_mount_0_1"),
    hand_view.dof_names.index("allegro_mount_1_2"),
    hand_view.dof_names.index("allegro_mount_2_3"),
    hand_view.dof_names.index("allegro_mount_3_4"),
    hand_view.dof_names.index("allegro_mount_4_5"),
    hand_view.dof_names.index("allegro_mount_5_6"),
  ]
  finger_indices = [hand_view.dof_names.index("index_joint_1"),
            hand_view.dof_names.index("index_joint_2"),
            hand_view.dof_names.index("index_joint_3"),
            hand_view.dof_names.index("middle_joint_1"),
            hand_view.dof_names.index("middle_joint_2"),
            hand_view.dof_names.index("middle_joint_3"),
            hand_view.dof_names.index("ring_joint_1"),
            hand_view.dof_names.index("ring_joint_2"),
            hand_view.dof_names.index("ring_joint_3"),
            hand_view.dof_names.index("thumb_joint_2"),
            hand_view.dof_names.index("thumb_joint_3"),
            ]
  demo_indices = [5, 6, 7,
                  9, 10, 11,
                  13, 14, 15,
                  3, 4]
  demo_angle_baseline = None
    
  world_t_grasp = se3.Transform(xyz=[0, 0, -0.4]) * world_t_initial
  initial_t_grasp = world_t_initial.inverse() * world_t_grasp
  grasp_duration = 5.0
  retract_duration = 3.0
  close_time = 5.0
  num_contacts = 0
  in_contact = False
  grasped = False
  retracting = False
#   print(wrist_indices)
#   print(finger_indices)
  time_elapsed = 0
  # import sys
  # sys.exit(1)
  while simulation_app.is_running() and time_elapsed < 15:
    world.step(render=True)

    frame_large = capture.read()
    if frame_large is None:
      continue
    if frame_large.shape[0] > frame_large.shape[1]:
      margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
      frame_large = frame_large[margin:-margin]
    else:
      margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
      frame_large = frame_large[:, margin:-margin]

    frame_large = np.flip(frame_large, axis=1).copy()
    frame = imresize(frame_large, (128, 128))

    _, theta_mpii = model.process(frame)
    demo_angles = []
    for quat in theta_mpii:
      demo_angles.append(np.degrees(np.arccos(np.dot(
        se3.Transform(rot=quat).matrix[:3, 0], np.array([1, 0, 0])))))
    # transform vs. no axis
    theta_mano = mpii_to_mano(theta_mpii)

    v = hand_mesh.set_abs_quat(theta_mano)
    v *= 2 # for better visualization
    v = v * 1000 + np.array([0, 0, 400])
    v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    # viewer.update_geometry(mesh)

    # viewer.poll_events()

    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_large, (window_size, window_size)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()

    # if keyboard.is_pressed("esc"):
    #   break

    clock.tick(30)

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

    if demo_angle_baseline is None:
      demo_angle_baseline = [demo_angles[demo_indice] for demo_indice in demo_indices]
    print([int(demo_angles[demo_indice]) for demo_indice in demo_indices])
    for i, index in enumerate(finger_indices):
        command[index] = np.radians(
          90 / 20 * abs(demo_angles[demo_indices[i]] - demo_angle_baseline[i]))

    command[thumb_joint_0_index] = 90.0
    hand_view.set_joint_position_targets(
      command
    )
    num_contacts = [hand_sensor.get_current_frame()["number_of_contacts"]
            for hand_sensor in hand_sensors]
    if in_contact:
      if time_elapsed - time_in_contact > 2.0:
        grasped = True
    else:
      if np.mean(num_contacts) > 5:
        in_contact = True
        print("-" * 10 + "in_contact")
        time_in_contact = time_elapsed
        actual_grasp_joint_vec = hand_view.get_joint_positions()[0][wrist_indices]
        initial_t_actual_grasp = se3.Transform(
          xyz=actual_grasp_joint_vec[:3], rot=actual_grasp_joint_vec[3:])
        world_t_actual_grasp = world_t_initial * initial_t_actual_grasp
    time_elapsed = time.time() - start_time


if __name__ == '__main__':
  run(OpenCVCapture())
  simulation_app.close()
