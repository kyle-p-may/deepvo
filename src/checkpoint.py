from datetime import datetime
import os
import torch

class Checkpointer:
  def __init__(self, path, load_loc):
    self.path = path
    self.load_loc = load_loc

  def CreateStringFromDate(self, date=datetime.now()):
    return "t" + str(date).replace(" ", "_")[:-7]

  def CreateDateFromString(self, string_rep):
    return datetime.strptime(string_rep, 't%Y-%m-%d_%H:%M:%S')
  
  def FindCheckpointFile(self, tag, i):
    f = os.path.join(self.path, tag + str(i) + '.pt')
    assert os.path.exists(f)
    return f

  def LoadFromCheckpoint(self, model, opt, tag, i):
    checkpoint_file = self.FindCheckpointFile(tag, i)

    if checkpoint_file == None:
      raise RunTimeError('no valid checkpoint files')
    
    assert os.path.exists(checkpoint_file), 'assuming that this file exists after finding it'

    checkpoint = torch.load(checkpoint_file, map_location=self.load_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']
  
  def CreateCheckpoint(self, model, opt, loss, epoch, tag, i):
    basename = tag + str(i)
    full_cp_path = os.path.join(self.path, basename + '.pt')

    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'opt_state_dict' : opt.state_dict(),
      'loss' : loss
    }, full_cp_path)