import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision

import poses as P

class VideoDataset(Dataset):

  class TrajectoryInfo:
    def __init__(self, seq, start, end):
      self.seq = seq
      self.start = start
      self.end = end

  def __init__(self, image_dir, pose_dir, seq_info, trajectory_length, transform=None):
    # Args:
    # image_dir is the directory that contains all image sequences
    # pose_dir is the directory that contains the pose information
    # seq_info contains the sequence tag (i.e. "00") and then the range of valid images
    #   The range is [start, end) and are integers
    # trajectory_length will be the length of each trajectory
    # transform will be applied to each image prior to creating
    # described stack

    # needs to maintain a list of all starting points
    # for each trajectory. Then, when __getitem__ is called,
    # return 
    self.image_dir = image_dir
    self.pose_dir = pose_dir
    self.traj_info = []
    self.transform = transform
    for seq, start, end in seq_info:
      for i in range(start, max(end+1-trajectory_length, start+1)):
        traj_end = min(i + trajectory_length, end + 1)
        self.traj_info.append( VideoDataset.TrajectoryInfo(seq, i, traj_end) )
    
  def __len__(self):
    return len(self.traj_info)
  
  def __getitem__(self, idx):
    traj = self.traj_info[idx]
    pose_file = os.path.join(self.pose_dir, traj.seq + '.txt')

    image_sequence = []    
    pose = []
    kitti_pose = []

    for img_index in range(traj.start, traj.end):
      basename = str(img_index).zfill(10) + '.png'
      full_image_path = os.path.join(self.image_dir, traj.seq, basename)
      assert os.path.exists(full_image_path), 'expecting that this photo exists'
      i = torchvision.io.read_image(full_image_path)
      norm_i = i / 255.0
      if self.transform:
        final_i = self.transform(norm_i)
      image_sequence.append( final_i.numpy() )

    assert os.path.exists(pose_file), 'expecting pose file to exist'
    with open(pose_file) as ip:
      lines = ip.readlines()[traj.start:traj.end]
      for line in lines:
        kitti = np.array([float(f) for f in line.split()])
        deepvo_pose = P.Poses.translate(kitti) 
        assert deepvo_pose.shape == (6,), 'expecting pose to have 6 elements'
        pose.append( deepvo_pose )
        kitti_pose.append( kitti )

    p = np.stack(pose) 
    kp = np.stack(kitti_pose)
    images = np.stack(image_sequence) 

    sample = {
      'sequence': torch.from_numpy(images),
      'poses': torch.from_numpy(p),
      'kitti_poses': torch.from_numpy(kp)
    }
    return sample