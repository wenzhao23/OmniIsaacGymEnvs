from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import pickle
import time
from typing import Any, Mapping

from data_types import geometry_utils
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
from omni.isaac.sensor import Camera
from omniisaacgymenvs.robots.articulations import a_sample
from omniisaacgymenvs.robots.articulations.views import a_sample_hand_view
from omniisaacgymenvs.utils.config_utils import sim_config
from omegaconf import DictConfig
import torch


def create_hand(stage, hand_prim_path):
  """Creates a hand from USD, linked with an already constructed hand type.
  """
  hand_start_translation = np.array([-0.05, 0.0, 0.4])
  hand_start_orientation = (
    se3.Transform(rot=np.radians([0, 105, 0])) *
    se3.Transform(rot=np.radians([0, 0, 0]))
  ).quaternion
  hand = a_sample.ASample(
    prim_path=hand_prim_path,
    name="hand",
    translation=hand_start_translation,
    orientation=hand_start_orientation,
  )
  # from omni.isaac.core.utils.prims import get_all_matching_child_prims
  # from omni.isaac.core.prims.rigid_prim import RigidPrim
  # all_prims = get_all_matching_child_prims(hand_prim_path, lambda _: True)
  # print(all_prims)
  hand.set_shadow_hand_properties(
    stage=stage, hand_prim=hand.prim)
  hand.set_motor_control_mode(
    stage=stage, hand_path="/World/aSampleForearm/aSampleForearm")

  hand_view = a_sample_hand_view.ASampleHandView(
    prim_paths_expr="/World/aSampleForearm/aSampleForearm",
    name="a_sample_hand_view")
  return hand, hand_view


def run():
  np.random.seed(17)

  # Creates the world  
  world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0}
  world = World(**world_settings)
  world.scene.add_default_ground_plane()

  # Adds objects to manipulate
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
  cube_support = DynamicCuboid(
      name=f"cube_support",
      position=np.array([0, 0, 0.15]),
      prim_path=f"/World/CubeSupport",
      scale=np.array([0.0515, 0.0515, 0.1015]),
      size=1.0,
      color=np.array([0, 0, 1]),
      physics_material=physics_material
    )
  world.scene.add(cube_support)
  cube_lift = DynamicCuboid(
      name=f"cube_lift",
      position=np.array([0, 0, 0.25]),
      prim_path=f"/World/CubeLift",
      scale=np.array([0.0515, 0.1515, 0.0515]),
      size=1.0,
      color=np.array([0, 0, 1]),
      physics_material=physics_material
    )
  world.scene.add(cube_lift)

  # Creates the hand
  stage = get_current_stage()
  hand_prim_path = "/World/aSampleForearm"
  hand, hand_view = create_hand(stage, hand_prim_path)
  world.scene.add(hand_view)

  # # Creates static camera
  # head_camera = Camera(
  #     prim_path="/World/camera",
  #     position=np.array([0.5, 0.0, 1.0]),
  #     frequency=20,
  #     resolution=(480, 640),
  #     orientation=(se3.Transform(rot=[0, 0, np.radians(180)]) *
  #                  se3.Transform(rot=[0, np.radians(50), 0])).quaternion,
  # )
  # head_camera.set_focal_length(1.0)
  # head_camera.set_focus_distance(1.0)
  # head_camera.set_horizontal_aperture(2.0955)
  # head_camera.set_vertical_aperture(1.52905)
  # head_camera.set_clipping_range(0.01, 10000)

  # # # Creates in-hand camera
  # hand_camera = Camera(
  #     prim_path="/World/kuka_allegro/kuka_allegro/palm_link/Camera",
  #     # position=np.array([0.0, 0.0, 0.0]),
  #     # orientation=[1, 0, 0, 0],
  #     frequency=20,
  #     resolution=(240, 320))
  # hand_camera.set_focal_length(0.5)
  # hand_camera.set_focus_distance(1.0)
  # hand_camera.set_horizontal_aperture(2.0955)
  # hand_camera.set_vertical_aperture(1.52905)
  # hand_camera.set_clipping_range(0.01, 10000)

  # Creates contact sensors on the hand
  hand_base_prim_path = "/World/aSampleForearm/aSampleForearm/"
  contact_links = [
    "Proximal_Asm__1",
    "Medial_Distal_Asm__1",
    "Proximal_Asm__4",
    "Medial_Distal_Asm__4",
    "Proximal_Asm__2",
    "Medial_Distal_Asm__2",
    "Proximal_Asm__3",
    "Medial_Distal_Asm__3",
    "Thumb_Proximal_2___Asm__1",
    "Thumb_Medial_Distal_Link_2___Asm__1",
  ]
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

  # # Inspects all hand joints
  # for prim in stage.TraverseAll():
  #   prim_type = prim.GetTypeName()
  #   if "Joint" in prim_type:
  #     print(prim)

  world.reset()
  # head_camera.initialize()
  # hand_camera.initialize()

  thumb_joint_0_index = hand_view.dof_names.index("THJ1")

  wrist_names = ["wrist_t_artificial_0",
                 "artificial_0_t_1",
                 "artificial_1_t_2",
                 "artificial_2_t_3",
                 "artificial_3_t_4",
                 "artificial_4_t_5"]
  wrist_indices = [hand_view.dof_names.index(name) for name in wrist_names]

  finger_names = ["THJ2", "THJ3", "FFJ1", "FFJ2", "FFJ3",
                  "MFJ2", "MFJ3", "RFJ2", "RFJ3", "LFJ2", "LFJ3"]
  finger_indices = [hand_view.dof_names.index(name) for name in finger_names]

  # Records the hand base pose and initial poses
  world_t_base_vec = hand_view.get_world_poses([0])
  world_t_initial = se3.Transform(
    xyz=world_t_base_vec[0][0], rot=world_t_base_vec[1][0])
  world_t_grasp = se3.Transform(xyz=[0, 0, -0.4]) * world_t_initial
  initial_t_grasp = world_t_initial.inverse() * world_t_grasp

  # Hardcodes a few constants to manually set the grasp/retract/contact behavior
  grasp_duration = 5.0
  retract_duration = 3.0
  close_time = 5.0

  # TODO: organize to state machine to replace these state flags
  in_contact = False
  grasped = False
  retracting = False

  time_elapsed = 0
  target_trajectory = []
  time_trajectory = []
  contact_trajectory = []
  start_time = time.time()
  tmp = True
  wait_for_recording = time.time()
  while simulation_app.is_running() and time_elapsed < 12:

    world.step(render=True)
    # print(hand_camera.get_current_frame())
    # if time_elapsed > 1.0 and tmp:
    #   from matplotlib import pyplot as plt
    #   imgplot = plt.imshow(hand_camera.get_rgba()[:, :, :3])
    #   plt.show()
    #   tmp = False
    
    # Intializes the hand joint command
    command = [0] * 18

    if grasped:

      if not retracting:

        # Configures right before retracting
        world_t_retract = se3.Transform(xyz=[0, 0, 0.4]) * world_t_actual_grasp
        actual_grasp_t_retract = world_t_actual_grasp.inverse() * world_t_retract
        actual_grasp_t_retract = initial_t_grasp.inverse()
        time_retract = time_elapsed
        finger_joint_grasp = hand_view.get_joint_positions()[0][finger_indices]
        retracting = True

      # Retracts to a predefined pose
      command[:3] = actual_grasp_joint_vec[:3] + actual_grasp_t_retract.translation * min(
        1, (time_elapsed - time_retract) / retract_duration)
      for i, index in enumerate(finger_indices):
        command[index] = finger_joint_grasp[i]

    else:

      # Moves downwards until contact
      if not in_contact:
        command[:3] = initial_t_grasp.translation * min(1, time_elapsed / grasp_duration)

      # Maintains palm position upon contact and only moves fingers
      else:
        command[:3] = actual_grasp_joint_vec[:3]

      # Starts to close fingers after the hardcoded close_time
      # TODO: this is a hack, when finger starts closing can be determined by a preshape
      # including a palm pose and finger initial configurations
      if time_elapsed > close_time:
        for i, index in enumerate(finger_indices):
          command[index] = -0.3 * (time_elapsed - close_time)

    # Hardcodes the thumb to be roughly antipodal
    command[thumb_joint_0_index] = np.radians(-90.0)

    # Sends hand joint commands
    hand_view.set_joint_position_targets(
      command
    )

    # Queries forward kinematics for all link poses
    joint_config = {}
    wrist_joint_vec = hand_view.get_joint_positions()[0][wrist_indices]
    for joint_name, joint_pos in zip(wrist_names, wrist_joint_vec):
      joint_config[joint_name] = joint_pos
    finger_joint_vec = hand_view.get_joint_positions()[0][finger_indices]
    for joint_name, joint_pos in zip(finger_names, finger_joint_vec):
      joint_config[joint_name] = joint_pos

    # Records the contact positions and normals
    all_contact_vecs = []
    target_object = "/World/CubeLift"
    world_t_target = cube_lift.get_world_pose()

    for hs in hand_sensors:

      # [(position, normal)]
      contact_vecs = []      
      frame = hs.get_current_frame()

      # Only grabs the contacts with the target object
      for contact in frame["contacts"]:
        if contact["body0"] == target_object:
          hand_contact = contact["body1"]
        elif contact["body1"] == target_object:
          print("TARGET OBJECT IS BODY1!")
          hand_contact = contact["body0"]
        else:
          continue
        link_name = hand_contact.split("/")[-1]

        world_t_link = se3.Transform()
        contact_vecs.append(
          (geometry_utils.transform_points(contact["position"].reshape(3, 1),
                                           world_t_link),
          geometry_utils.rotate_points(contact["normal"].reshape(3, 1),
                                       world_t_link)),
        )
      all_contact_vecs.append(contact_vecs)

      print(frame)
    contact_trajectory.append(all_contact_vecs)

    # TOOD: this is a hack determining contact state depending on the average number of contacts
    num_contacts = [len(hand_sensor.get_current_frame()["contacts"])
            for hand_sensor in hand_sensors]
    num_contacts = [1]
    if in_contact:

      # Hardcodes continuing closing fingers for 5.0 seconds after in_contact
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

    # # Gets force torque sensor readings
    # print(np.linalg.norm(
    #   hand_view._physics_view.get_force_sensor_forces()[0, :, :], axis=1))

    time_elapsed = time.time() - start_time
    time_trajectory.append(frame["time"])
    target_trajectory.append(world_t_target)
    print('-' * 30)

  # Saves the contact trajectory
  data = {
    'time': time_trajectory,
    'contact': contact_trajectory,
    'target': target_trajectory
  }
  with open(os.path.expanduser("~/data/grasping/contacts.pkl"), "wb") as f:
    pickle.dump(data, f)

if __name__ == '__main__':
  run()
  simulation_app.close()
