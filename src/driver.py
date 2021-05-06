import datetime
import torch
import numpy as np

from model import DeepVOModel
from device import DeviceHelper
from pretrained import ImportPretrainedModel
from checkpoint import Checkpointer
from optfactory import OptimizerFactory
from dataset import RandomInMemoryVideoDataset

class Driver:
  def evaluate(model, v_dl, epoch):
    outputs = [model.validation_step(batch) for batch in v_dl]
    epoch_loss = torch.stack(outputs).mean().item()
    print ('Epoch: ' + str(epoch) + ' -> Loss: ' + str(epoch_loss))
    return epoch_loss

  def fit(epochs, model, optimizer, params, cp):
    history = []

    for epoch in range(epochs):
      t_dl, v_dl = RandomInMemoryVideoDataset.DataLoaderFactory(params)
      print('Epoch[' + str(epoch) + ']: ' + str(datetime.datetime.now())) 
      model.train()
      for batch in t_dl:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      
      model.eval()
      with torch.no_grad():
        result = Driver.evaluate(model, v_dl, epoch)
        history.append( (epoch, result) )

      cp.CreateCheckpoint(model, optimizer, epoch, None, params.checkpoint_tag, epoch)
    
    return history
  
  def CreateModelAndOpt(params, dh, cp):
    assert params.load_pretrained or params.load_cp, 'should use pretrained or cp'

    model = DeepVOModel(params)

    if params.load_pretrained:
      ImportPretrainedModel(model, params.pretrained_path, dh.device)
      opt = OptimizerFactory.create(params.opt_type, params.lr, model)
    else:
      # load checkpoint
      opt = OptimizerFactory.create(params.opt_type, params.lr, model)
      cp.LoadFromCheckpoint(model, opt, params.checkpoint_tag, params.checkpoint_i)

    if torch.cuda.is_available():
      model.cuda()

    return model, opt
  
  def evalOnVideo(model, ds, outputfile):
    
    # just assume that this is the base pose for now
    predicted_rel_poses = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

    model.eval()
    with torch.no_grad():
      for i in range(len(ds)):
        if i % 200 == 0:
          print('At frame ' + str(i) + ' of ' + str(len(ds)))
        stacked_frame, _ = ds[i]

        stacked_frame.cuda()
        assert stacked_frame.is_cuda, 'tensor should be on gpu'
        prediction = model.eval_forward(stacked_frame).squeeze().cpu()
        predicted_rel_poses.append( prediction.numpy() )

      pred = np.stack(predicted_rel_poses)
      pred = pred[1:, :] + pred[:-1, :] # calculate the absolutes
      np.save(outputfile, pred)