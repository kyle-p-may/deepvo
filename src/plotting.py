import matplotlib.pyplot as plt
import numpy as np
import os

class Plotting:
  def getXY(poses):
    x_idx = 3
    y_idx = 5
    x = [v for v in poses[:, x_idx]]
    y = [v for v in poses[:, y_idx]]
    return x, y

  def loadPoses(gt_dir, pred_dir, run_prefix, seq):
    gt_filename = os.path.join(gt_dir, seq + '_poses.npy')
    pred_filename = os.path.join(pred_dir, run_prefix + seq + '.npy')
    gt = np.load(gt_filename)
    pred = np.load(pred_filename)
    return gt, pred

  def plot(gt_dir, pred_dir, run_prefix, out_dir, seqs):
    p_color = 'r'
    g_color = 'b'

    for seq in seqs:
      gt, pred = Plotting.loadPoses(gt_dir, pred_dir, run_prefix, seq)
      gX, gY = Plotting.getXY(gt)
      pX, pY = Plotting.getXY(pred)

      plt.clf()
      plt.plot(gX, gY, color = g_color, label='Ground Truth')
      plt.plot(pX, pY, color = p_color, label='DeepVO')
      plt.gca().set_aspect('equal', adjustable='datalim')
      plt.legend()
      plt.title('Sequence ' + seq)
      
      savename = os.path.join(out_dir, run_prefix + seq + '_route.png')
      plt.savefig(savename)