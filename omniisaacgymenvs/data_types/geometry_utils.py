import numpy as np

from omniisaacgymenvs.data_types import se3


def transform_points(points: np.ndarray, ref_t_cur: se3.Transform):
  return np.dot(ref_t_cur.matrix[:3, :3], points) + ref_t_cur.matrix[:3, 3:]

def rotate_points(points: np.ndarray, ref_t_cur: se3.Transform):
  return np.dot(ref_t_cur.matrix[:3, :3], points)
