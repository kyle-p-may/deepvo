import torch.nn as nn

from parameters import *

class ConvLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
    super().__init__()
    self.layer = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Dropout(dropout)
    )
  
  def forward(self, xb):
    return self.layer(xb)

class ConvLayerWithBatchNorm(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
    super().__init__()
    self.layer = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=True),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Dropout(dropout)
    )
  
  def forward(self, xb):
    return self.layer(xb)

def ConvFactory(batch_norm, in_channels, out_channels, kernel_size, stride, dropout):
  if batch_norm:
    return ConvLayerWithBatchNorm(in_channels, out_channels, kernel_size, stride, dropout)
  else:
    return ConvLayer(in_channels, out_channels, kernel_size, stride, dropout)

class DeepVOModel(nn.Module):
  def __init__(self, parameters):
    super().__init__()
    
    self.network0 = nn.Sequential(
      ConvFactory(parameters.batch_norm, parameters.channels[0], parameters.channels[1], kernel_size=parameters.kernel_size[0], stride=parameters.stride[0], dropout=parameters.conv_dropout[0]),
      ConvFactory(parameters.batch_norm, parameters.channels[1], parameters.channels[2], kernel_size=parameters.kernel_size[1], stride=parameters.stride[1], dropout=parameters.conv_dropout[1]),
      ConvFactory(parameters.batch_norm, parameters.channels[2], parameters.channels[3], kernel_size=parameters.kernel_size[2], stride=parameters.stride[2], dropout=parameters.conv_dropout[2]),
      ConvFactory(parameters.batch_norm, parameters.channels[3], parameters.channels[4], kernel_size=parameters.kernel_size[3], stride=parameters.stride[3], dropout=parameters.conv_dropout[3]),
      ConvFactory(parameters.batch_norm, parameters.channels[4], parameters.channels[5], kernel_size=parameters.kernel_size[4], stride=parameters.stride[4], dropout=parameters.conv_dropout[4]),
      ConvFactory(parameters.batch_norm, parameters.channels[5], parameters.channels[6], kernel_size=parameters.kernel_size[5], stride=parameters.stride[5], dropout=parameters.conv_dropout[5]),
      ConvFactory(parameters.batch_norm, parameters.channels[6], parameters.channels[7], kernel_size=parameters.kernel_size[6], stride=parameters.stride[6], dropout=parameters.conv_dropout[6]),
      ConvFactory(parameters.batch_norm, parameters.channels[7], parameters.channels[8], kernel_size=parameters.kernel_size[7], stride=parameters.stride[7], dropout=parameters.conv_dropout[7]),
      ConvFactory(parameters.batch_norm, parameters.channels[8], parameters.channels[9], kernel_size=parameters.kernel_size[8], stride=parameters.stride[8], dropout=parameters.conv_dropout[8])
    )
    self.rnn = nn.LSTM(
        input_size=parameters.rnn_input_size,
        hidden_size=parameters.rnn_hidden_size,
        num_layers=parameters.rnn_num_layers,
        dropout=parameters.rnn_internal_dropout,
        batch_first=True
    )
    self.rnn_dropout = nn.Dropout(parameters.rnn_dropout)
    self.linear = nn.Linear(in_features=parameters.rnn_hidden_size, out_features=6)
  
  def forward(self, x):
    # this will be {batch, seq, channel, width, height}
    # and we want to concatenate along channel
    batch_size = x.size(0)
    seq_len = x.size(1)
    x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
    x = self.network0(x)
    x = x.view(batch_size, seq_len, -1)

    x, _ = self.rnn(x)
    x = self.rnn_dropout(x)
    return self.linear(x)

  def eval_forward(self, x):
    # this will be {batch, seq, channel, width, height}
    # and we want to concatenate along channel
    batch_size = x.size(0)
    seq_len = x.size(1)
    x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
    x = self.network0(x)
    x = x.view(batch_size, seq_len, -1)

    x, _ = self.rnn(x)
    x = self.rnn_dropout(x)
    return self.linear(x)

  
  def loss(self, predicted, ground_truth):
    # both predicted and ground truth have the following structure
    # {batch, traj, 6}
    assert predicted.shape == ground_truth.shape, 'expecting shape to be same for loss computation'

    angle_loss = nn.functional.mse_loss(predicted[:, :, :3], ground_truth[:,:,:3])
    pos_loss = nn.functional.mse_loss(predicted[:,:,3:], ground_truth[:,:,3:])

    kappa = 100

    return kappa * angle_loss + pos_loss
  
  def training_step(self, batch):
    images, poses = batch
    predictions = self(images)
    training_loss = self.loss(predictions.float(), poses.float())
    return training_loss
  
  def validation_step(self, batch):
    images, poses = batch
    out = self(images)
    val_loss = self.loss(out.float(), poses.float())
    return val_loss