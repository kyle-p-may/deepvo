import datetime
import torch
import numpy as np

from model import DeepVOModel
from device import DeviceHelper
from pretrained import ImportPretrainedModel
from checkpoint import Checkpointer
from optfactory import OptimizerFactory

class Driver:
  def evaluate(model, v_dl, epoch):
    outputs = [model.validation_step(batch) for batch in v_dl]
    epoch_loss = torch.stack(outputs).mean().item()
    print ('Epoch: ' + str(epoch) + ' -> Loss: ' + str(epoch_loss))
    return epoch_loss

  def fit(epochs, model, t_dl, v_dl, optimizer):
    history = []

    for epoch in range(epochs):
      print('Epoch[' + str(epoch) + ']: ' + str(datetime.datetime.now())) 
      model.train()
      for batch in t_dl:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      
      model.eval()
      result = evaluate(model, v_dl, epoch)
      history.append( (epoch, result) )
    
    return history
  
  def CreateModelAndOpt(params, dh, cp):
    assert params.load_pretrained or params.load_cp, 'should use pretrained or cp'

    model = DeepVOModel(params)

    if params.load_pretrained:
      ImportPretrainedModel(model, params.pretrained_path, dh.device)
    else:
      # load checkpoint
      cp.LoadFromCheckpoint(model, opt)

    if torch.cuda.is_available():
      model.cuda()

    opt = OptimizerFactory.create(params.opt_type, params.lr, model)

    return model, opt
  
  def evalOnVideo(model, ds, outputfile):
    
    predicted_poses = []
    model.eval()
    with torch.no_grad():
      for i in range(len(ds)):
        if i % 200 == 0:
          print('At frame ' + str(i) + ' of ' + str(len(ds)))
        stacked_frame, _, _ = ds[i]

        stacked_frame.cuda()
        assert stacked_frame.is_cuda, 'tensor should be on gpu'
        prediction = model.eval_forward(stacked_frame).squeeze().cpu()
        predicted_poses.append( prediction.numpy() )

      pred = np.stack(predicted_poses)
      np.save(outputfile, pred)