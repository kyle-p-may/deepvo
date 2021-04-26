from datetime import datetime
import os
import torch

class Checkpointer:
  def __init__(self, path, load_loc):
    self.path = path
    self.load_loc = load_loc

  def CreateStringFromDate(self, date=datetime.now()):
    return str(date.replace(" ", "_"))

  def CreateDateFromString(self, string_rep):
    return datetime.strptime(string_rep, '%Y-%m-%d_%H:%M:%S.%f')
  
  # this returns the filename (full path)
  # of the most recent checkpoint
  def FindMostRecentCheckpoint(self):
    files = os.listdir(self.path)

    dates = []
    for f in files:
      full_file_path = os.path.join(self.path, f)
      # first, make sure it actually is a file
      if os.path.isfile(full_file_path):
        split = os.path.splitext(f)
        date, ext = split[0], split[1]
        if ext == '.pt':
          try:
            # if the filename is poorly formatted this
            # will throw an exception
            dates.append( CreateDateFromString(date) )
          except:
            print('Poorly formatted checkpoint filename: ' + full_file_path)
            print('Continuing')
            continue
      else:
        continue
  
    if len(dates) == 0:
      return None
    else:
      # find largest and then return the filename
      max_date = max(dates)
      date_string = CreateStringFromDate(max_date)
      return os.path.join(self.path, date_string + '.pt')
  
  def LoadFromCheckpoint(self, model, opt):
    checkpoint_file = FindMostRecentCheckpoint()

    if checkpoint_file == None:
      raise RunTimeError('no valid checkpoint files')
    
    assert os.path.exists(checkpoint_file), 'assuming that this file exists after finding it'

    checkpoint = torch.load(checkpoint_file, map_location=self.load_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoints['opt_state_dict'])

    return checkpoint['epochs'], checkpoints['loss']
  
  def CreateCheckpoint(self, model, opt, loss, epoch):
    basename = CreateStringFromDate()    
    full_cp_path = os.path.join(self.path, basename + '.pt')

    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'opt_state_dict' : opt.state_dict(),
      'loss' : loss
    }, full_cp_path)