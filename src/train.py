from driver import *
from parameters import *

from device import DeviceHelper
from checkpoint import Checkpointer

from dataset import VideoDataset

dh = DeviceHelper()
cp = Checkpointer(params.checkpoint_path, dh.device)

# load model and optimizer
model, opt = Driver.CreateModelAndOpt(params, dh, cp)

# run training
history = Driver.fit(params.epochs, model, opt, params, cp)

# save history and model to checkpoint
with open(params.log_file, 'a') as of:
  of.write('*' * 50)
  of.write('Loss sequence')
  of.write(str(history))

epoch, loss = zip(*history)

cp.CreateCheckpoint(model, opt, params.epochs, min(loss), params.checkpoint_tag, params.epochs)