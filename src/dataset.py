import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
import random

import poses as P
import preprocessing as Prep
from device import DeviceDataLoader

class Helper:
  def createFrameName(img_idx):
    return str(img_idx).zfill(10) + '.npy'

  def loadConsecutiveFrames(path, seq, start, end):
    image_sequence = []
    for img_index in range(start, end+1):
      basename = Helper.createFrameName(img_index)
      full_image_path = os.path.join(path, seq, basename)
      assert os.path.exists(full_image_path), 'expecting that this photo exists'
      i = np.load(full_image_path)
      image_sequence.append( i )

    return np.stack(image_sequence)
  
  def selectRange(start, end, length):
    range_end = max(end-length, start)
    idx = random.randint(start, range_end)
    return idx, min(idx + length, end), min(end-start, length)
  
# this will provide an interface to an entire video
class SingleVideo(Dataset):
  def __init__(self, image_dir, pose_dir, seq_info):
    super().__init__()
    self.image_dir = image_dir
    self.pose_dir = pose_dir
    self.seq, self.start, self.end = seq_info

    full_pose_name = os.path.join(pose_dir, self.seq + '_poses.npy')
    raw_poses = np.load(full_pose_name)
    self.poses = raw_poses[:, :6]

  def __len__(self):
    return (self.end - self.start + 1) - 1
  
  def __getitem__(self, idx):
    img_idx = idx + self.start
    path = os.path.join(self.image_dir, self.seq, Helper.createFrameName(img_idx))
    next_path = os.path.join(self.image_dir, self.seq, Helper.createFrameName(img_idx))

    i = torch.from_numpy(np.load(path))
    next_i = torch.from_numpy(np.load(path))
    p = torch.from_numpy(self.poses[idx, :])

    im = torch.cat((next_i, i))
    im = im[None, None, :, :, :]
    return (im, p)

class RandomInMemoryVideoDataset:
  def DataLoaderFactory(params):
    training_ds = RandomInMemoryVideoDataset(
      params.image_dir,
      params.pose_dir,
      params.training_sequences,
      params.training_traj_length
    )

    eval_ds = RandomInMemoryVideoDataset(
      params.image_dir,
      params.pose_dir,
      params.eval_sequences,
      params.eval_traj_length
    )
    
    t_dl = DataLoader(training_ds, batch_size=params.batch_size, shuffle=True, pin_memory=True)
    e_dl = DataLoader(eval_ds, batch_size=params.batch_size*2, pin_memory=True)
    return DeviceDataLoader(t_dl), DeviceDataLoader(e_dl)

  def __init__(self, image_dir, pose_dir, seq_info, trajectory_length, transform=None):
    self.length = 1200
    super().__init__()

    si = random.choice(seq_info)
    self.seq, self.start, self.end = si
    self.traj_length = trajectory_length
    self.tl = self.traj_length

    self.start, self.end, self.length = Helper.selectRange(self.start, self.end, self.length+1)
    self.length = self.length - trajectory_length

    self.image_dir = image_dir
    self.pose_dir = pose_dir

    pose_file = os.path.join(self.pose_dir, self.seq + '_poses.npy') 

    # find the relative poses
    combined_poses = np.load(pose_file)
    pose_start = self.start - si[1]
    pose_end = self.end - si[1] + 1
    poses = combined_poses[pose_start:pose_end, :6]
    self.poses = torch.from_numpy(poses[1:, :] - poses[:-1, :])

    images = Helper.loadConsecutiveFrames(image_dir, self.seq, self.start, self.end)
    self.sequence = torch.from_numpy(images)

    assert self.sequence.shape[0] == self.poses.shape[0] + 1, 'poses and seq should be same'

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    end = idx + self.tl - 1
    stack = torch.cat((self.sequence[idx+1:end+1], self.sequence[idx:end]), dim=1)
    return stack, self.poses[idx:end]

class VideoDataset(Dataset):

  def DataLoaderFactory(params):
    training_ds = VideoDataset(
      params.image_dir,
      params.pose_dir,
      params.training_sequences,
      params.training_traj_length
    )

    eval_ds = VideoDataset(
      params.image_dir,
      params.pose_dir,
      params.eval_sequences,
      params.eval_traj_length
    )
    
    t_dl = DataLoader(training_ds, batch_size=params.batch_size, shuffle=True, pin_memory=True)
    e_dl = DataLoader(eval_ds, batch_size=params.batch_size*2, pin_memory=True)

    return DeviceDataLoader(t_dl), DeviceDataLoader(e_dl)

  class TrajectoryInfo:
    def __init__(self, seq, start, end):
      self.seq = seq
      self.start = start
      self.end = end

  def __init__(self, image_dir, pose_dir, seq_info, trajectory_length, transform=None):
    super().__init__()
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
    self.seq_info = {}
    self.seq_poses = {}
    for seq, start, end in seq_info:
      self.seq_info[seq] = (start, end)
      pose_file = os.path.join(self.pose_dir, seq + '_poses.npy') 

      pose = []
      combined_poses = np.load(pose_file)

      self.seq_poses[seq] = combined_poses[:, :6]
      
      for i in range(start, max(end+1-trajectory_length, start+1)):
        traj_end = min(i + trajectory_length, end + 1)
        self.traj_info.append( VideoDataset.TrajectoryInfo(seq, i, traj_end) )

  def __len__(self):
    return len(self.traj_info)

  def __getitem__(self, idx):
    traj = self.traj_info[idx]

    base_img_num, _ = self.seq_info[traj.seq]
    pose_start = traj.start - base_img_num
    pose_end = traj.end - base_img_num + 1
    p = self.seq_poses[traj.seq][pose_start : pose_end, :]
    p = p[1:, :] - p[:-1, :]

    images = Helper.loadConsecutiveFrames(self.image_dir, traj.seq, traj.start, traj.end)

    sample = {
      'sequence': torch.from_numpy(images),
      'poses': torch.from_numpy(p)
    }

    sample['sequence'] = torch.cat((sample['sequence'][:-1], sample['sequence'][1:]), dim=1)
    return (sample['sequence'], sample['poses'])