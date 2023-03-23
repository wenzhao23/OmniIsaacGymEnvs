from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from omni.isaac.cloner import GridCloner
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.prims import get_prim_at_path
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
  def __init__(self, name: str, offset: Optional[np.ndarray] = None):

    BaseTask.__init__(self, name=name, offset=offset)

    self._physics_material = PhysicsMaterial("/World/PhysicsMaterial")
    self._cameras = []
    self._hand_views = {}
    self._wrist_joint_indices = {}
    self._finger_joint_indices = {}

    # Hardcodes the thumb to be roughly antipodal,
    # which requires a sepcial query for thumb_joint_0
    self._thumb_joint_0_index = {}

    self._env_spacing = 0.5
    self._cloner = GridCloner(spacing=self._env_spacing)
    self._cloner.define_base_env("/World/kuka_allegro")

  @property
  def offset(self):
    return self._offset

  def post_reset(self) -> None:
    """Calls while doing a .reset() on the world.
    """
    for camera in self._camera_names:
      camera.initialize()
    
    for name, hand_view in self._hand_views.items():
      self._thumb_joint_0_index[
        name] = hand_view.dof_names.index("thumb_joint_0")

      wrist_names = ["allegro_mount_0_1",
                    "allegro_mount_1_2",
                    "allegro_mount_2_3",
                    "allegro_mount_3_4",
                    "allegro_mount_4_5",
                    "allegro_mount_5_6"]
      self._wrist_joint_indices[name] = [
        hand_view.dof_names.index(name) for name in wrist_names]

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
      self._finger_joint_indices[name] = [
        hand_view.dof_names.index(name) for name in finger_names]

  def setup_scene(self, scene):

    super().set_up_scene(scene)

    # Adds the ground
    scene.add_default_ground_plane()

    # Adds hand
    self._add_hand(scene)

    # Adds cameras
    self._add_static_camera()
    self._add_hand_camera()

    # Adds a supporting cube
    self._add_cube_support(scene)

    # Adds objects to manipulate
    self._add_cube_target(scene)

    # Shifts task objects to different positions for different environments
    self._move_task_objects_to_their_frame()

  def get_observations(self):
    pass

  def is_done(self):
    pass

  def _add_cube_target(self, scene):
    cube_target_prim_path = find_unique_string_name(
      initial_name="/World/CubeTarget",
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    cube_target_name = find_unique_string_name(
      initial_name="cube_target",
      is_unique_fn=lambda x: not self.scene.object_exists(x))

    cube_target = DynamicCuboid(
        name=cube_target_name,
        position=np.array([0, 0, 0.4]),
        prim_path=cube_target_prim_path,
        scale=np.array([0.0515, 0.1515, 0.0515]),
        size=1.0,
        color=np.array([0, 0, 1]),
        physics_material=self._physics_material
      )
    scene.add(cube_target)
    self._task_objects[cube_target_name] = cube_target

  def _add_cube_support(self, scene):
    cube_support_prim_path = find_unique_string_name(
      initial_name="/World/CubeSupport",
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    cube_support_name = find_unique_string_name(
      initial_name="cube_support",
      is_unique_fn=lambda x: not self.scene.object_exists(x))

    cube_support = DynamicCuboid(
        name=cube_support_name,
        position=np.array([0, 0, 0.3]),
        prim_path=cube_support_prim_path,
        scale=np.array([0.0515, 0.0515, 0.1015]),
        size=1.0,
        color=np.array([0, 0, 1]),
        physics_material=self._physics_material
      )
    scene.add(cube_support)
    self._task_objects[cube_support_name] = cube_support

  def _add_hand(self, scene):

    stage = get_current_stage()

    usd_path = "/World/kuka_allegro"
    prim_path = find_unique_string_name(
      initial_name=usd_path,
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    prim_name = prim_path.replace(usd_path, "hand")

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

    self._hand_view = shadow_hand_view.ShadowHandView(
      prim_paths_expr=prim_path + "/kuka_allegro",
      name=prim_name + "_view")

    scene.add(self._hand_view)
    self._task_objects[prim_name] = hand
    self._hand_name = prim_name
    self._hand_views["right"] = self._hand_view

  def _add_static_camera(self):
    # Creates static camera
    usd_path = "/World/camera"
    prim_path = find_unique_string_name(
      initial_name=usd_path,
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    prim_name = prim_path.replace(usd_path, "static_camera")

    # prim = get_prim_at_path(prim_path)
    # if not prim.IsValid():
    #     prim = define_prim(prim_path, "Xform")
    #     prim.GetReferences().AddReference(usd_path)

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
    usd_path = "/World/kuka_allegro/kuka_allegro/palm_link/Camera"
    prim_path = find_unique_string_name(
      initial_name=usd_path,
      is_unique_fn=lambda x: not is_prim_path_valid(x))
    prim_name = prim_path.replace(usd_path, "hand_camera")

    # prim = get_prim_at_path(prim_path)
    # if not prim.IsValid():
    #     prim = define_prim(prim_path, "Xform")
    #     prim.GetReferences().AddReference(usd_path)

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
