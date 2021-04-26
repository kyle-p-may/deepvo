import torch

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
    self.log_file = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\data\\history.txt'
    self.image_dir = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\data\\sequences'
    self.pose_dir = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\data\\poses'
    self.pretrained_path = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\model\\pretrained\\t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train'
    self.checkpoint_path = 'C:\\Users\\kylep\\Documents\\projects\\deepvo\\model'

    self.training_sequences = [('00', 0, 4540), ('02', 0, 4660), ('08', 1100, 5170), ('09', 0, 1590)]
    self.eval_sequences = [('01', 0, 1100), ('05', 0, 2760), ('06', 0, 1100), ('07', 0, 1100), ('10', 0, 1200)]

    self.training_traj_length = 5
    self.eval_traj_length = 10
    self.epochs = 1
    self.batch_size = 1

    self.load_pretrained = True
    self.load_cp = False

    self.torch_location = torch.device('cuda')

    self.opt_type = 'Adagrad'
    self.lr = 0.005
  
  def print(self):
    print('Current Hyperparameter Values')
    temp = vars(self)
    for i in temp:
      print(i, ':', temp[i])

params = NewDeepVOParameters()
params.print()