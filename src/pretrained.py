import torch

def ImportPretrainedModel(model, path, location):

  ## define a mapping from the pretrained model
  ## to my model
  pretrained_to_model = {
    'conv1.0' : 'network.0.layer.0',
    'conv1.1' : 'network.0.layer.1',
    'conv2.0' : 'network.1.layer.0',
    'conv2.1' : 'network.1.layer.1',
    'conv3.0' : 'network.2.layer.0',
    'conv3.1' : 'network.2.layer.1',
    'conv3_1.0' : 'network.3.layer.0',
    'conv3_1.1' : 'network.3.layer.1',
    'conv4.0' : 'network.4.layer.0',
    'conv4.1' : 'network.4.layer.1',
    'conv4_1.0' : 'network.5.layer.0',
    'conv4_1.1' : 'network.5.layer.1',
    'conv5.0' : 'network.6.layer.0',
    'conv5.1' : 'network.6.layer.1',
    'conv5_1.0' : 'network.7.layer.0',
    'conv5_1.1' : 'network.7.layer.1',
    'conv6.0' : 'network.8.layer.0',
    'conv6.1' : 'network.8.layer.1',
    'rnn' : 'network.9',
    'linear' : 'network.11'
  }

  # build reverse map
  model_to_pretrained = {}
  for key, value in pretrained_to_model.items():
    model_to_pretrained[value] = key
  
  pretrained_model = torch.load(path, map_location=location)

  sd = model.state_dict()

  # for each key in my model,
  # I will check to see if there are weights available in
  # the pretrained version
  # To do this, we find the translated name
  # and then see if its in the loaded state dictionary 
  # Then, add it to a new state dictionary that I
  # can use to modify the model

  def GetStateParts(key):
    last_period = key.rfind('.')
    return key[:last_period], key[last_period+1:]
  
  def CombineStateParts(first,name):
    return first + '.' + name 

  new_state_dict = {}

  for key in sd.keys():
    key_root, key_name = GetStateParts(key)
    if key_root in model_to_pretrained:
      source = model_to_pretrained[key_root]
      source_key = CombineStateParts(source, key_name)      

      if source_key in pretrained_model:
        print('Mapping ' + source_key + ' to ' + key)
        new_state_dict[key] = pretrained_model[source_key]
      else:
        print('Maintaining default for ' + key)
        new_state_dict[key] = sd[key]
    else:
      print('Maintaining default for ' + key)
      new_state_dict[key] = sd[key]

  model.load_state_dict(new_state_dict)