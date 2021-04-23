class NewDeepVOParameters:
  def __init__(self):
    self.batch_norm = True

    self.channels = [6, 64, 128, 256, 256, 512, 512, 512, 512, 1024]
    self.kernel_size = [7, 5, 5, 3, 3, 3, 3, 3, 3]
    self.conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5]
    
    self.rnn_internal_dropout = 0
    self.rnn_dropout = 0.5
    self.rnn_hidden_size = 1000
    self.rnn_num_layers = 2

    self.img_width = 1280
    self.img_height = 384