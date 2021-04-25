import numpy as np
from scipy.spatial.transform import Rotation as R

## Poses are stored in length 6 vector
## with the euler angles first (xyz) and then
## the position
class Poses:
  def translateFromKittiToDeepvo(pose):
    assert (pose.shape == (12,)), "input pose correct shape"

    # last column is the current position
    position = np.array([pose[3], pose[7], pose[11]])

    transformation_mat = pose.reshape(3, 4)[:, :-1]
    r = R.from_matrix(transformation_mat)
    eulers = r.as_euler('xyz')

    return np.concatenate((eulers, position))
  
  def translate(pose):
    both = Poses.translateFromKittiToDeepvo(pose)
    return both[0:6]