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
    
    self.network = nn.Sequential(
      ConvFactory(parameters.batch_norm, parameters.channels[0], parameters.channels[1], kernel_size=parameters.kernel_size[0], stride=parameters.stride[0], dropout=parameters.conv_dropout[0]),
      ConvFactory(parameters.batch_norm, parameters.channels[1], parameters.channels[2], kernel_size=parameters.kernel_size[1], stride=parameters.stride[1], dropout=parameters.conv_dropout[1]),
      ConvFactory(parameters.batch_norm, parameters.channels[2], parameters.channels[3], kernel_size=parameters.kernel_size[2], stride=parameters.stride[2], dropout=parameters.conv_dropout[2]),
      ConvFactory(parameters.batch_norm, parameters.channels[3], parameters.channels[4], kernel_size=parameters.kernel_size[3], stride=parameters.stride[3], dropout=parameters.conv_dropout[3]),
      ConvFactory(parameters.batch_norm, parameters.channels[4], parameters.channels[5], kernel_size=parameters.kernel_size[4], stride=parameters.stride[4], dropout=parameters.conv_dropout[4]),
      ConvFactory(parameters.batch_norm, parameters.channels[5], parameters.channels[6], kernel_size=parameters.kernel_size[5], stride=parameters.stride[5], dropout=parameters.conv_dropout[5]),
      ConvFactory(parameters.batch_norm, parameters.channels[6], parameters.channels[7], kernel_size=parameters.kernel_size[6], stride=parameters.stride[6], dropout=parameters.conv_dropout[6]),
      ConvFactory(parameters.batch_norm, parameters.channels[7], parameters.channels[8], kernel_size=parameters.kernel_size[7], stride=parameters.stride[7], dropout=parameters.conv_dropout[7]),
      ConvFactory(parameters.batch_norm, parameters.channels[8], parameters.channels[9], kernel_size=parameters.kernel_size[8], stride=parameters.stride[8], dropout=parameters.conv_dropout[8]),
      nn.LSTM(
        input_size=parameters.rnn_input_size,
        hidden_size=parameters.rnn_hidden_size,
        num_layers=parameters.rnn_num_layers,
        dropout=parameters.rnn_internal_dropout,
        batch_first=True
      ),
      nn.Dropout(parameters.rnn_dropout),
      nn.Linear(in_features=parameters.rnn_hidden_size, out_features=6)
    )
  
  def forward(self, xb):
    return self.network(xb)