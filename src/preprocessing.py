import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms

from poses import Poses
from parameters import params

class CenterToZeroAndSubtractMean(object):
  def __init__(self, channel_means):
    self.channel_means = channel_means
    
  def __call__(self, pic):
    num_channels = self.channel_means.shape[0]
    assert num_channels == pic.shape[0], 'expecting same number of channels'
    for c in range(num_channels):
      pic[c, :, :] = pic[c, :, :] - 0.5 - self.channel_means[c]
    
    return pic

    
class Preprocessor:
  def LoadMeans(mean_file):
    return np.load(mean_file)

  def TransformFactory(means):
    return transforms.Compose([
            CenterToZeroAndSubtractMean(means),
            transforms.Resize((params.img_height, params.img_width))
          ])

  def DetermineMeanRGB(image_dir, sequence_list, output_file):
    ## this could overflow, so might need to come back to this eventually
    sum = np.array([0., 0., 0.])
    count = 0
    for seq in sequence_list:
      seq_dir = os.path.join(image_dir, seq)
      if not os.path.exists(seq_dir):
        print('Invalid sequence number, directory not found: ' + seq)
        continue

      image_list = os.listdir(seq_dir)

      for image in image_list:
        full_image_path = os.path.join(seq_dir, image)
        i = torchvision.io.read_image(full_image_path)
        norm_i = i / 255.0

        sums = np.sum(norm_i.numpy(), (1,2)) / (i.shape[1] * i.shape[2])
        assert sums.shape == (3,)

        sum += sums
        count += 1

    sum = sum / count
    np.save(output_file, sum)
    return sum
      
  def ProcessGroundTruthPoses(pose_dir, sequence_list, output_dir):

    def createOutputFilePath(odir, s):
      return os.path.join(odir, s + '_poses.npy')

    def createInputFilePath(pdir, s):
      return os.path.join(pdir, s + '.txt')

    if not os.path.exists(output_dir):
      print ('Output directory did not exist')
      os.makedirs(output_dir)

    for seq in sequence_list:
      print('Processing ' + seq)

      outputfile = createOutputFilePath(output_dir, seq)
      inputfile = createInputFilePath(pose_dir, seq)

      if not os.path.exists(inputfile):
        print('Input file ' + inputfile + ' not found, continuing to next seq')
        continue
      
      with open(inputfile) as inputstream:
        lines = inputstream.readlines()
        all_poses = np.array()
        for line in lines:
          kitti_pose = [float(f) for f in line.split()]
          kitti_pose = np.array(kitti_pose)
          assert(kitti_pose.shape == (12,)), 'expecting kitti pose to length 12'
          deepvo_pose = Poses.translateFromKittiToDeepvo(kitti_pose)

          combined_poses = np.concatenate(deepvo_pose, kitti_pose)
          all_poses = np.concatenate(all_poses, combined_poses)

        all_poses.reshape((-1, 18))
        np.save(output_file, all_poses)
        print('Saved output at ' + output_file)

  def ValidFile(sequence_number, photo_number):
    valid_ranges = {
      '00': ('0000000000', '0000004540'),
      '01': ('0000000000', '0000001100'),
      '02': ('0000000000', '0000004660'),
      '03': ('0000000000', '0000000800'),
      '04': ('0000000000', '0000000270'),
      '05': ('0000000000', '0000002760'),
      '06': ('0000000000', '0000001100'),
      '07': ('0000000000', '0000001100'),
      '08': ('0000001100', '0000005170'),
      '09': ('0000000000', '0000001590'),
      '10': ('0000000000', '0000001200')
    }

    assert sequence_number in valid_ranges.keys(), 'invalid sequence number'
    minimum, maximum = valid_ranges[sequence_number]

    assert(len(minimum) == 10), 'minimum length expected to be 10'
    assert(len(maximum) == 10), 'maximum_length expected to be 10'
    assert(len(photo_number) == 10), 'photo_number length expected to be 10'

    return photo_number >= minimum and photo_number <= maximum

  ## this function will remove any files that 
  def CleanupFiles(image_dir, sequence_list):
    print('hello')
    for seq in sequence_list:
      assert (len(seq) == 2), 'expecting sequence to be length 2'
      print('Checking sequence ' + seq)

      seq_dir = os.path.join(image_dir, seq)
      if not os.path.exists(seq_dir):
        print('Invalid sequence number, directory not found: ' + seq)
        continue

      image_list = os.listdir(seq_dir)
      for image in image_list:
        image_path = os.path.join(seq_dir, image)
        assert os.path.exists(image_path), 'image should exist at this point'
        photo_number = os.path.splitext(os.path.basename(image))[0]
        assert( len(photo_number) == 10 ), 'expecting length 10 photo number'

        if not Preprocessor.ValidFile(seq, photo_number):
          os.remove(image_path)
          print('Removing ' + image_path)