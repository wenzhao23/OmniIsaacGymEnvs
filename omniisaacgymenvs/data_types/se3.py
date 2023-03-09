"""Define SE3 Transform.
"""
import numpy as np
from typing import Union, Sequence, Text, Tuple

from omniisaacgymenvs.data_types import transformations as tr


VectorType = Union[np.ndarray, Sequence]


class Transform(object):
  """SE3 Transformation, uniquely defined by a 4x4 matrix.
  """

  def __init__(self, matrix: np.ndarray = None,
               xyz: VectorType = None,
               rot: VectorType = None):
    """Constructs the SE3 transform.

    Via matrix OR xyz and/or rot(quaternion or euler angles
    in radians).
    """
    if matrix is not None:
      assert xyz is None and rot is None
      self._matrix = matrix
      self._xyz = tr.translation_from_matrix(self._matrix)
      self._wxyz = tr.quaternion_from_matrix(self._matrix)

    else:
      xyz = [0, 0, 0] if xyz is None else xyz
      rot = [1, 0, 0, 0] if rot is None else rot
      self._xyz = np.array(xyz)
      if len(rot) == 3:
        self._wxyz = tr.quaternion_from_euler(rot[0], rot[1], rot[2])
      else:
        assert len(rot) == 4
        self._wxyz = np.array(rot) / np.linalg.norm(rot)
      r, p, y = self.rpy
      self._matrix = tr.euler_matrix(r, p, y)
      for i in range(3):
        self._matrix[i, 3] = self._xyz[i]

  @property
  def quaternion(self) -> np.ndarray:
    return self._wxyz

  @property
  def translation(self) -> np.ndarray:
    return self._xyz

  @property
  def rpy(self) -> np.ndarray:
    return np.array(tr.euler_from_quaternion(self._wxyz, 'sxyz'))

  @property
  def matrix(self) -> np.ndarray:
    return self._matrix

  def inverse(self) -> 'Transform':
    return Transform(matrix=np.linalg.inv(self._matrix))

  def to_list(self) -> Sequence:
    res = [e for e in self._xyz]
    res.extend([e for e in self._wxyz])
    return res

  def __mul__(self, other: 'Transform') -> 'Transform':
    """Multiplies another Transform"""
    return Transform(matrix=np.dot(self.matrix, other.matrix))

  def __repr__(self) -> Text:
    """xyz, rpy representation"""
    return "<Translation: {} Rotation: {}".format(
      self.translation, self.rpy)


def add(first: Transform, second: Transform, weight_first: float = 0.5):
  translation = weight_first * first.translation + (
    1 - weight_first) * second.translation
  rotation = tr.quaternion_slerp(
    first.quaternion, second.quaternion, fraction=1 - weight_first)
  return Transform(xyz=translation, rot=rotation)


def rotation_to_axis_angle(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
  """Convert the rotation matrix into the axis-angle notation.

  Conversion equations
  ====================
  From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix),

  Args:
    matrix:  np.array, 3x3
  Returns:
    The 3D rotation axis and angle.
  """
  # Axes.
  axis = np.zeros((3,))
  axis[0] = matrix[2, 1] - matrix[1, 2]
  axis[1] = matrix[0, 2] - matrix[2, 0]
  axis[2] = matrix[1, 0] - matrix[0, 1]

  # Angle.
  r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
  t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
  theta = np.arccos((t - 1) / 2)

  # Normalise the axis.
  if r != 0:
    axis = axis / r

  # Return the data.
  return axis, theta


# ---------from/to Tranforms-----------------
# def kdl_to_transform(kdl_frame):
#   quat = kdl_frame.M.GetQuaternion()
#   trans = kdl_frame.p

#   return Transform(xyz=trans, rot=quat)


def msg_to_transform(pose_msg):
  xyz = [pose_msg.position.x,
         pose_msg.position.y,
         pose_msg.position.z]

  quat = [pose_msg.orientation.w,
          pose_msg.orientation.x,
          pose_msg.orientation.y,
          pose_msg.orientation.z]

  return Transform(xyz=xyz, rot=quat)


# def transform_to_msg(transform):
#   pose_msg = geometry_msgs.msg.Pose()
#   pose_msg.position.x = transform.translation[0]
#   pose_msg.position.y = transform.translation[1]
#   pose_msg.position.z = transform.translation[2]
#   pose_msg.orientation.x = transform.quaternion[0]
#   pose_msg.orientation.y = transform.quaternion[1]
#   pose_msg.orientation.z = transform.quaternion[2]
#   pose_msg.orientation.w = transform.quaternion[3]

#   return pose_msg


# def xyzrpy_to_transform(xyzrpy):
#   return Transform(
#     xyz=[xyzrpy.position.x, xyzrpy.position.y, xyzrpy.position.z],
#     rot=[xyzrpy.rpy.roll, xyzrpy.rpy.pitch, xyzrpy.rpy.yaw])
