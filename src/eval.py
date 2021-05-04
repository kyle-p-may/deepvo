import argparse
import os

from driver import *
from parameters import *
from device import *
from checkpoint import Checkpointer

from dataset import SingleVideo
from plotting import Plotting

def evaluateVideos(seqs, tag):
  dh = DeviceHelper()
  print ('Using ' + str(dh.device))
  cp = Checkpointer(params.checkpoint_path, dh.device)
  model, _ = Driver.CreateModelAndOpt(params, dh, cp)

  for seq_info in seqs:
    seq, _, _ = seq_info
    print ('Processing ' + seq)
    s = SingleVideo(params.image_dir, params.pose_dir, seq_info)
    ds = DeviceDataset(s)
    output = os.path.join(params.pred_dir, tag + '_' + seq + '.npy')
    Driver.evalOnVideo(model, ds, output)
  
  seq_list = [x for x, _, _ in seqs]

  Plotting.plot(params.pose_dir, params.pred_dir, tag + '_', params.result_dir, seq_list)

def initCLI():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--seq', action='append', required=True)
  parser.add_argument('-t' ,'--tag', required=True)
  return parser

def seqToSeqInfo(p, s):
  all_seqs = p.training_sequences + p.eval_sequences
  return [x for x in all_seqs if x[0] in s]

if __name__ == '__main__':
  import sys
  parser = initCLI()
  args = parser.parse_args(sys.argv[1:])
  evaluateVideos(seqToSeqInfo(params, args.seq), args.tag)