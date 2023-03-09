import numpy as np

from data_types import se3


def test_se3():
  pose_1 = se3.Transform(xyz=[1, 2, 3])
  pose_0 = se3.Transform(matrix=pose_1.matrix)
  assert np.allclose((pose_1 * pose_0.inverse()).matrix, np.eye(4))

  pose_2 = se3.Transform(rot=[0, 0, np.pi / 2])
  pose_3 = pose_2 * pose_1
  expected = [-2, 1, 3] + pose_2.quaternion.tolist()
  assert all([abs(a - b) < 1e-4 for a, b in zip(
    pose_3.to_list(), expected)])

  pose_4 = se3.Transform(rot=[0, 0, -np.pi / 2])
  should_be_identity = (pose_2 * pose_4).matrix
  assert np.linalg.norm(should_be_identity - np.eye(4)) < 1e-4

  assert str(pose_1)


def test_add():
  pose_1 = se3.Transform(xyz=[1, 2, 3])
  pose_2 = se3.Transform(xyz=[-1, -2, -3])
  pose_added = se3.add(pose_1, pose_2)
  assert np.allclose(pose_added.to_list(), [0, 0, 0, 1, 0, 0, 0])
  pose_added = se3.add(pose_1, pose_2, 0.75)
  assert np.allclose(pose_added.to_list(), [0.5, 1, 1.5, 1, 0, 0, 0])

  pose_1 = se3.Transform(rot=[0, 0, 0])
  pose_2 = se3.Transform(rot=np.radians([180, 0, 0]))
  pose_added = se3.add(pose_1, pose_2)
  expected = se3.Transform(rot=np.radians([90, 0, 0]))
  assert np.allclose(pose_added.to_list(), expected.to_list())
  pose_added = se3.add(pose_1, pose_2, 0.75)
  expected = se3.Transform(rot=np.radians([45, 0, 0]))
  assert np.allclose(pose_added.to_list(), expected.to_list())


def test_rotation_to_axis_angle():
  pose = se3.Transform(rot=[0, 0, np.pi / 2])
  axis, angle = se3.rotation_to_axis_angle(pose.matrix)
  assert np.all(axis == np.array([0, 0, 1]))
  assert abs(angle - np.pi / 2) < 1e-4
  pose = se3.Transform(rot=[0, 0, 0])
  axis, angle = se3.rotation_to_axis_angle(pose.matrix)
  assert angle == 0


if __name__ == "__main__":
    test_rotation_to_axis_angle()
    test_se3()
    test_add()
