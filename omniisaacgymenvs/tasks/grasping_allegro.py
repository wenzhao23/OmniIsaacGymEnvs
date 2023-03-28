from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from omni.isaac.cloner import GridCloner
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.prims import get_all_matching_child_prims
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.sensor import Camera
from omni.isaac.sensor import ContactSensor
from omniisaacgymenvs.data_types import se3
from omniisaacgymenvs.robots.articulations import shadow_hand
from omniisaacgymenvs.robots.articulations.views import shadow_hand_view


class Grasping(ABC, BaseTask):
  """A task that consists of a hand, sensors, and objects in the scene.
  """
  def __init__(self, name: str, world, offset: Optional[np.ndarray] = None):

    BaseTask.__init__(self, name=name, offset=offset)

    self._world = world
    self._physics_material = PhysicsMaterial("/World/PhysicsMaterial")
    self._cameras = []
    self._hand_views = {}
    self._wrist_joint_indices = {}
    self._finger_joint_indices = {}

    # Hardcodes the thumb to be roughly antipodal,
    # which requires a sepcial query for thumb_joint_0
    self._thumb_joint_0_index = {}

    self._num_envs = 100
    self._env_spacing = 1.0
    self._cloner = GridCloner(spacing=self._env_spacing, num_per_row=10)
    self.default_zero_env_path = "/World/envs/env_0"
    self.default_base_env_path = "/World/envs"
    self._cloner.define_base_env(self.default_base_env_path)
    define_prim(self.default_zero_env_path)

  @property
  def offset(self):
    return self._offset

  def post_reset(self) -> None:
    """Calls while doing a .reset() on the world.
    """

    print("!" * 100)
    print(self._hand_view.get_world_poses())

    for camera in self._camera_names:
      camera.initialize()
    
    # for name, hand_view in self._hand_views.items():
    #   self._thumb_joint_0_index[
    #     name] = hand_view.dof_names.index("thumb_joint_0")

    #   wrist_names = ["allegro_mount_0_1",
    #                 "allegro_mount_1_2",
    #                 "allegro_mount_2_3",
    #                 "allegro_mount_3_4",
    #                 "allegro_mount_4_5",
    #                 "allegro_mount_5_6"]
    #   self._wrist_joint_indices[name] = [
    #     hand_view.dof_names.index(name) for name in wrist_names]

    #   finger_names = ["thumb_joint_0",
    #                   "thumb_joint_2",
    #                   "thumb_joint_3",
    #                   "index_joint_1",
    #                   "index_joint_2",
    #                   "index_joint_3",
    #                   "middle_joint_1",
    #                   "middle_joint_2",
    #                   "middle_joint_3",
    #                   "ring_joint_1",
    #                   "ring_joint_2",
    #                   "ring_joint_3"]
    #   self._finger_joint_indices[name] = [
    #     hand_view.dof_names.index(name) for name in finger_names]

  def setup_scene(self, scene):

    super().set_up_scene(scene)

    collision_filter_global_paths = list()

    # Adds the ground
    self._ground_plane_path = "/World/defaultGroundPlane"
    scene.add_default_ground_plane(prim_path=self._ground_plane_path)
    collision_filter_global_paths.append(self._ground_plane_path)

    # Adds hand
    self._add_hand(scene)

    # Adds cameras
    self._add_static_camera()
    self._add_hand_camera()

    # Adds a supporting cube
    self._add_cube_support(scene)

    # Adds objects to manipulate
    self._add_cube_target(scene)

    # # Shifts task objects to different positions for different environments
    # self._move_task_objects_to_their_frame()

    prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
    self._env_pos = self._cloner.clone(
      source_prim_path="/World/envs/env_0",
      prim_paths=prim_paths,
      replicate_physics=False)
    # self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
    self._cloner.filter_collisions(
        self._world.get_physics_context().prim_path,
        "/World/collisions",
        prim_paths,
        collision_filter_global_paths)
    # self.set_initial_camera_params(
      # camera_position=[10, 10, 3], camera_target=[0, 0, 0])

    self._hand_view = shadow_hand_view.ShadowHandView(
      prim_paths_expr="/World/envs/env_.*/kuka_allegro",
      name="all_hands_view")
    scene.add(self._hand_view)

    hand_start_translation = np.array([-0.1, 0.0, 0.6])
    hand_start_orientation = (
      se3.Transform(rot=np.radians([0, 90, 0])) *  # 120
      se3.Transform(rot=np.radians([0, 0, 140]))
    ).quaternion
    shifted_positions = np.array(self._env_pos) + hand_start_translation.reshape((1, 3))
    shifted_orientations = np.zeros((self._num_envs, 4)) + hand_start_orientation.reshape((1, 4))
    self._hand_view.set_world_poses(shifted_positions, shifted_orientations)

    # Randomizes all targets' sizes
    all_prims = get_all_matching_child_prims("/World/envs", lambda x: "CubeTarget" in x)
    for prim in all_prims:
      RigidPrim(prim_path=prim.GetPrimPath()).set_local_scale(np.random.uniform(low=[0.03, 0.03, 0.03], high=[0.2, 0.2, 0.2]))

    self._hand_views["right"] = self._hand_view

  def get_observations(self):
    pass

  def is_done(self):
    pass

  def _add_cube_target(self, scene):
    # cube_target_prim_path = find_unique_string_name(
    #   initial_name="/World/envs/env_0/CubeTarget",
    #   is_unique_fn=lambda x: not is_prim_path_valid(x))
    cube_target_prim_path = "/World/envs/env_0/CubeTarget"
    cube_target_name = find_unique_string_name(
      initial_name="cube_target",
      is_unique_fn=lambda x: not self.scene.object_exists(x))

    cube_target = DynamicCuboid(
        name=cube_target_name,
        translation=np.array([0, 0, 0.4]),
        prim_path=cube_target_prim_path,
        scale=np.array([0.05, 0.15, 0.05]),
        size=1.0,
        color=np.array([0, 0, 1]),
        physics_material=self._physics_material
      )
    scene.add(cube_target)
    self._task_objects[cube_target_name] = cube_target

  def _add_cube_support(self, scene):
    cube_support_prim_path = "/World/envs/env_0/CubeSupport"
    cube_support_name = find_unique_string_name(
      initial_name="cube_support",
      is_unique_fn=lambda x: not self.scene.object_exists(x))

    cube_support = DynamicCuboid(
        name=cube_support_name,
        translation=np.array([0, 0, 0.2]),
        prim_path=cube_support_prim_path,
        scale=np.array([0.05, 0.05, 0.10]),
        size=1.0,
        color=np.array([0, 0, 1]),
        physics_material=self._physics_material
      )
    scene.add(cube_support)
    self._task_objects[cube_support_name] = cube_support

  def _add_hand(self, scene):

    stage = get_current_stage()

    initial_prim_path = "/World/envs/env_0"
    prim_path = initial_prim_path
    # prim_path = find_unique_string_name(
    #   initial_name=initial_prim_path,
    #   is_unique_fn=lambda x: not is_prim_path_valid(x))
    prim_name = prim_path.replace("/", "_") + "hand"

    hand_start_translation = np.array([-0.1, 0.0, 0.6])
    hand_start_orientation = (
      se3.Transform(rot=np.radians([0, 90, 0])) *  # 120
      se3.Transform(rot=np.radians([0, 0, 140]))
    ).quaternion
    hand = shadow_hand.ShadowHand(
      prim_path=prim_path,
      name=prim_name,
      translation=hand_start_translation,
      orientation=hand_start_orientation,
    )
    hand.set_shadow_hand_properties(
      stage=stage, shadow_hand_prim=hand.prim)
    hand.set_motor_control_mode(
      stage=stage, shadow_hand_path=hand.prim_path)

    self._task_objects[prim_name] = hand
    self._hand_name = prim_name


  def _add_static_camera(self):
    # Creates static camera
    usd_path = "/World/envs/env_0/camera"
    prim_path = find_unique_string_name(
      initial_name=usd_path,
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    prim_name = prim_path.replace(usd_path, "static_camera")

    head_camera = Camera(
        prim_path=prim_path,
        name=prim_name,
        position=np.array([0.5, 0.0, 1.0]),
        frequency=20,
        resolution=(480, 640),
        orientation=(se3.Transform(rot=[0, 0, np.radians(180)]) *
                    se3.Transform(rot=[0, np.radians(50), 0])).quaternion,
    )
    head_camera.set_focal_length(1.0)
    head_camera.set_focus_distance(1.0)
    head_camera.set_horizontal_aperture(2.0955)
    head_camera.set_vertical_aperture(1.52905)
    head_camera.set_clipping_range(0.01, 10000)
    self._cameras.append(head_camera)

    # Adds camera to _task_objects as its parent is /World
    self._task_objects[prim_name] = head_camera

  def _add_hand_camera(self):
    # Creates in-hand camera
    # usd_path = "/World/kuka_allegro/kuka_allegro/palm_link/Camera",
    usd_path = "/World/envs/env_0/kuka_allegro/palm_link/Camera"
    prim_path = find_unique_string_name(
      initial_name=usd_path,
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    prim_name = prim_path.replace(usd_path, "hand_camera")

    hand_camera = Camera(
        prim_path=prim_path,
        name=prim_name,
        frequency=20,
        resolution=(240, 320))
    hand_camera.set_focal_length(0.5)
    hand_camera.set_focus_distance(1.0)
    hand_camera.set_horizontal_aperture(2.0955)
    hand_camera.set_vertical_aperture(1.52905)
    hand_camera.set_clipping_range(0.01, 10000)

    self._cameras.append(hand_camera)

  def _add_contact_sensor(self, scene):
    # Creates contact sensors on the hand
    hand_base_prim_path = self._task_objects[self._hand_name
                                             ].prim_path + "/kuka_allegro/"
    contact_links = [
      "index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip",
      "index_link_1", "index_link_2", "index_link_3",
      "middle_link_1", "middle_link_2", "middle_link_3",
      "ring_link_1", "ring_link_2", "ring_link_3",
      "thumb_link_1", "thumb_link_2", "thumb_link_3"]
    self._hand_sensors = []
    for contact_link in contact_links: 
      self._hand_sensors.append(scene.add(
        ContactSensor(
          prim_path=hand_base_prim_path + contact_link + "/contact_sensor",
          name="contact_sensor_" + contact_link,
          min_threshold=0,
          max_threshold=10000000,
          radius=0.01,
        )
      ))
      self._hand_sensors[-1].add_raw_contact_data_to_frame()


# # Another way to attach fix the hand's base joint
    # print("!" * 100)
    # stage = get_current_stage()
    # # # env_pos = stage.GetPrimAtPath(
    # # #   f"{self.default_base_env_path}/env_{0}").GetAttribute(
    # # #   "xformOp:translate").Get()
    # from pxr import UsdPhysics, Gf
    # anchor_pos = Gf.Vec3f(0, 0, 0) + Gf.Vec3f(-0.1, 0.0, 0.6)
    # self.fix_to_ground(stage,
    #                    self.default_base_env_path + "/env_0/kuka_allegro/ground_t_hand",
    #                    "/World/envs/env_0" + "/kuka_allegro/allegro_mount",
    #                    anchor_pos)

  # def fix_to_ground(self, stage, joint_path, prim_path, anchor_pos):
  #   from pxr import UsdPhysics, Gf
  #   # D6 fixed joint
  #   d6FixedJoint = UsdPhysics.Joint.Define(stage, joint_path)
  #   d6FixedJoint.CreateBody0Rel().SetTargets([self._ground_plane_path])
  #   d6FixedJoint.CreateBody1Rel().SetTargets([prim_path])
  #   print(anchor_pos)
  #   d6FixedJoint.CreateLocalPos0Attr().Set(anchor_pos)
  #   d6FixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
  #   d6FixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
  #   d6FixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
  #   # lock all DOF (lock - low is greater than high)
  #   d6Prim = stage.GetPrimAtPath(joint_path)
  #   limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
  #   limitAPI.CreateLowAttr(1.0)
  #   limitAPI.CreateHighAttr(-1.0)
  #   limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
  #   limitAPI.CreateLowAttr(1.0)
  #   limitAPI.CreateHighAttr(-1.0)
  #   limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
  #   limitAPI.CreateLowAttr(1.0)
  #   limitAPI.CreateHighAttr(-1.0)