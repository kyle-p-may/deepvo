import torch.optim

class OptimizerFactory:
  def create(type, lr, model):
    if type == 'Adagrad':
      return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
      raise ValueError('Unsupported optimizer type')