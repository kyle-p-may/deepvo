class NewDeepVOParameters:
  def __init__(self):
    self.batch_norm = True

    self.channels = [6, 64, 128, 256, 256, 512, 512, 512, 512, 1024]
    self.kernel_size = [7, 5, 5, 3, 3, 3, 3, 3, 3]
    self.conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5]
    self.stride = [2, 2, 2, 1, 2, 1, 2, 1, 2]
    
    self.rnn_internal_dropout = 0
    self.rnn_dropout = 0.5
    self.rnn_hidden_size = 1000
    self.rnn_num_layers = 2
    # TODO: this is just a magic number that I found
    self.rnn_input_size = 30720

    self.img_width = 608
    self.img_height = 184

    self.mean_file = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\data\\rgb_means.npy'
    self.image_dir = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\data\\sequences'
    self.pose_dir = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\data\\poses'
    self.pretrained_path = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\model\\pretrained\\t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train'

    self.training_sequences = [('00', 0, 4540), ('02', 0, 4660), ('09', 0, 1590)]

    self.training_traj_length = 17
    self.epochs = 128
    self.batch_size = 32

params = NewDeepVOParameters()