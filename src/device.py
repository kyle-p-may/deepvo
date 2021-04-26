import torch

class DeviceHelper:
  def __init__(self):
    self.device = self.get_default_device()

  def get_default_device(self):
    if torch.cuda.is_available():
      return torch.device('cuda')
    else:
      return torch.device('cpu')
  
  def to_device(self, data):
    if isinstance(data, (list, tuple)):
      return [self.to_device(x) for x in data]
    return data.to(self.device, non_blocking=True)
  

class DeviceDataLoader:
  def __init__(self, dl):
    self.dh = DeviceHelper()
    self.dl = dl
    
  def __iter__(self):
    for b in self.dl:
      yield self.dh.to_device(b)
  
  def __len__(self):
    return len(self.dl)