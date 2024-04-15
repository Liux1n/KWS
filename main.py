# Copyright (C) 2022 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


import torch
import dataset
import os
import time
import math
import random
import wandb  
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchsummary import summary
# from model import DSCNNS, DSCNNM, DSCNNL
from dscnn import DSCNNS, DSCNNM, DSCNNL
from utils import remove_txt, parameter_generation, load_config
from copy import deepcopy
from pthflops import count_ops
from train import Train
import argparse

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print ('Torch Version:', torch.version.__version__)
print('Device:', device)
config = load_config("config.yaml")

parser = argparse.ArgumentParser()

parser.add_argument('--task', 
                  choices=['dil_task_0', 'dil_task_1' , 'dil_task_2', 'dil_task_3' , 'dil_joint',
                           'cil_task_0', 'cil_task_1', 'cil_task_2', 'cil_task_3', 'cil_joint'
                           ], 
                  default='dil_task_0',
                  required=True,
                  help='training mode (default: dil_task_0)')

parser.add_argument('--method', 
                  choices=['base','source_only', 'finetune', 'joint', 'ER_random', 'ECBRS', 'custum'
                           ], 
                  default='base',
                  required=True,
                  help='training  method (default: base)')

parser.add_argument('--noise_mode', choices=['nlkws', 'nakws', 'odda'], default='nakws',
                    help='noise mode')
parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
parser.add_argument('--wandb', action='store_true', help="Enable wandb log.")
parser.add_argument('--pretrain', action='store_true', help="Enable pretrain.")
parser.add_argument('--background_volume', type=int, help="Set background volume.")
args = parser.parse_args()


if args.wandb:
  wandb.init(
            project=config['project_name'], 
            entity="liux1n", 
            config=config
            )

if args.task == 'base':
   print('Base model')
if args.debug:
  print('Debug mode, 1 epoch.')

# Parameter generation
training_parameters, data_processing_parameters = parameter_generation(args, config)  # To be parametrized

# if args.method == 'ER_random' or 'ECBRS' or 'custum':
#    training_parameters['batch_size'] = config['new_data_ratio']*training_parameters['batch_size']

# Dataset generation
audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

train_size = audio_processor.get_size('training')
valid_size = audio_processor.get_size('validation')
test_size = audio_processor.get_size('testing')
print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

# # Model generation and analysis
# if args.task == 'dil_task_0' or args.task == 'dil_task_1' or \
#    args.task == 'dil_task_2' or args.task == 'dil_task_3' or args.task == 'dil_joint':
#   n_classes = config['n_classes'] + 2 # 35 words + 1 unknown + 1 silence

if args.task == 'cil_task_0' and args.method != 'joint': # 1 to 17
  n_classes = 17 + 2 # 17 words + 1 unknown + 1 silence
  # n_classes = 35 + 2 

elif args.task == 'cil_task_1' and args.method != 'joint': # 1 to 23
  n_classes = 23 + 2 # 6 words + 1 unknown + 1 silence
  # n_classes = 35 + 2 

elif args.task == 'cil_task_2' and args.method != 'joint': # 1 to 29
  n_classes = 29 + 2 # 6 words + 1 unknown + 1 silence
  # n_classes = 35 + 2 

elif (args.task == 'cil_task_3' or args.task == 'cil_joint') and args.method != 'joint': # 1 to 35
  n_classes = 35 + 2 # 6 words + 1 unknown + 1 silence
  # n_classes = 35 + 2
else: 
  n_classes = config['n_classes'] + 2 # 35 words + 1 unknown + 1 silence



model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
model.to(device)
summary(model,(1,49,data_processing_parameters['feature_bin_count']))
dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
# count_ops(model, dummy_input)

# Training initialization
training_environment = Train(audio_processor, training_parameters, model, device, args, config)

# Removing stored inputs and activations
remove_txt()

print('Method:', args.method)
pre_train = args.pretrain
if pre_train:
  print('Task '+args.task)
  # Task 0. Train from scratch
  # start=time.clock_gettime(0)
  start=time.process_time()
  
  training_environment.train(model)
  print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
else:
  print('Task: '+args.task)
  start=time.process_time()
  # Load pretrained model 
  # DIL models:
  # model_name = 'base_dil_task_0_model.pth'
  # model_name = 'base_dil_joint_model.pth'
  # model_name = 'base_12_dil_task_0_model.pth'
  # fine-tune models:
  # model_name = 'finetune_dil_task_1_model.pth'
  # random ER models:
  # model_name = 'ER_random_dil_task_1_model.pth'
  # model_name = 'ER_random_dil_task_2_model.pth'

  # DIL vol=9 models:
  # model_name = 'base20240414-231812dil_task_0_model.pth'
  
  # CIL models:
  # model_name = 'base_cil_task_0_model.pth'
  # model_name = 'ER_random_cil_task_1_model.pth'
  # model_name = 'ER_random_cil_task_2_model.pth'
  model_name = 'joint_12_cil_joint_model.pth'
  
  model_path = './models/' + model_name 
  # Load the state dict

  if args.task == 'dil_task_0' or args.task == 'dil_task_1' or args.task == 'dil_task_2' \
  or args.task == 'dil_task_3' or args.task == 'dil_joint' or args.task == 'cil_task_0' \
  or args.task == 'cil_joint':
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
  else:
    if args.method == 'joint':
      model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    else:

      # cil_task_0: n_classes = 19
      # cil_task_1: n_classes = 25
      # cil_task_2: n_classes = 31 
      # cil_task_3: n_classes = 37
      # Load the state dict
      state_dict = torch.load(model_path, map_location=torch.device('cuda'))

      # Get the weights and biases of the old layer
      old_weights = state_dict['fc1.weight']
      old_bias = state_dict['fc1.bias']

      # Create new weights and biases with the desired size
      new_weights = torch.randn([n_classes, 64])
      new_bias = torch.randn([n_classes])
      if args.method == 'source_only' or args.method == 'base' or args.method == 'finetune':
        n_classes_prev = 19
      elif args.method == 'ER_random':
        if args.task == 'cil_task_1':
          n_classes_prev = 19
        elif args.task == 'cil_task_2':
          n_classes_prev = 25
        elif args.task == 'cil_task_3':
          n_classes_prev = 31
      # Copy the old weights and biases into the new ones
      new_weights[:n_classes_prev] = old_weights
      new_bias[:n_classes_prev] = old_bias

      # Replace the weights and biases in the state dict
      state_dict['fc1.weight'] = new_weights
      state_dict['fc1.bias'] = new_bias

      # Load the modified state dict into the model
      model.load_state_dict(state_dict)
      # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

    
  print('Load model from: ' + model_path)

  if args.method == 'source_only':
    print('Using source only model. No continual training needed.')
  elif args.method == 'joint':
    print('Using joint-training model. No continual training needed.')
  else: 
    
    if args.method == 'finetune':
      training_environment.continual_train(model)
    elif args.method == 'ER_random':

      # task 1:
      print('ER_random')

      if args.task == 'dil_task_1':
        print('Starting ER_random on dil_task_1')
        # initialize memory buffer
        memory_buffer = {}
        # input shape # ( 1, 49, 10])
        # label shape # (1,)
        memory_buffer['inputs'] = []
        memory_buffer['labels'] = []
        memory_buffer = training_environment.er_random_train(model,memory_buffer, base = True)

      elif args.task == 'dil_task_2':
        print('Starting ER_random on dil_task_2')
        # load memory buffer from ./buffer/dil_task_1_buffer_input
        buffer_path_input = './buffer/dil_task_1_buffer_input.npy'
        buffer_path_label = './buffer/dil_task_1_buffer_label.npy'
        memory_buffer_inputs = np.load(buffer_path_input)
        memory_buffer_labels = np.load(buffer_path_label)
        memory_buffer = {}
        memory_buffer['inputs'] = memory_buffer_inputs
        memory_buffer['labels'] = memory_buffer_labels

        memory_buffer = training_environment.er_random_train(model,memory_buffer, base = False)
      elif args.task == 'dil_task_3':
        print('Starting ER_random on dil_task_3')
        # load memory buffer from ./buffer/dil_task_2_buffer_input
        buffer_path_input = './buffer/dil_task_2_buffer_input.npy'
        buffer_path_label = './buffer/dil_task_2_buffer_label.npy'
        memory_buffer_inputs = np.load(buffer_path_input)
        memory_buffer_labels = np.load(buffer_path_label)
        memory_buffer = {}
        memory_buffer['inputs'] = memory_buffer_inputs
        memory_buffer['labels'] = memory_buffer_labels
        memory_buffer = training_environment.er_random_train(model,memory_buffer, base = False)

      # CIL tasks
      elif args.task == 'cil_task_1':
        print('Starting ER_random on cil_task_1')
        # initialize memory buffer
        memory_buffer = {}
        # input shape # ( 1, 49, 10])
        # label shape # (1,)
        memory_buffer['inputs'] = []
        memory_buffer['labels'] = []
        memory_buffer = training_environment.er_random_train(model,memory_buffer, base = True)

      elif args.task == 'cil_task_2':
        print('Starting ER_random on cil_task_2')
        # load memory buffer from ./buffer/dil_task_1_buffer_input
        buffer_path_input = './buffer/cil_task_1_buffer_input.npy'
        buffer_path_label = './buffer/cil_task_1_buffer_label.npy'
        memory_buffer_inputs = np.load(buffer_path_input)
        memory_buffer_labels = np.load(buffer_path_label)
        memory_buffer = {}
        memory_buffer['inputs'] = memory_buffer_inputs
        memory_buffer['labels'] = memory_buffer_labels

        memory_buffer = training_environment.er_random_train(model,memory_buffer, base = False)
      elif args.task == 'cil_task_3':
        print('Starting ER_random on cil_task_3')
        # load memory buffer from ./buffer/dil_task_2_buffer_input
        buffer_path_input = './buffer/cil_task_2_buffer_input.npy'
        buffer_path_label = './buffer/cil_task_2_buffer_label.npy'
        memory_buffer_inputs = np.load(buffer_path_input)
        memory_buffer_labels = np.load(buffer_path_label)
        memory_buffer = {}
        memory_buffer['inputs'] = memory_buffer_inputs
        memory_buffer['labels'] = memory_buffer_labels
        memory_buffer = training_environment.er_random_train(model,memory_buffer, base = False)

      memory_buffer_inputs = memory_buffer['inputs']
      memory_buffer_labels = memory_buffer['labels']
      # convert to npy
      memory_buffer_inputs = np.array(memory_buffer_inputs.cpu()) 
      memory_buffer_labels = np.array(memory_buffer_labels.cpu())



      input_name = args.task + '_buffer_input.npy'
      label_name = args.task + '_buffer_label.npy'
      buffer_path_input = './buffer/' + input_name
      buffer_path_label = './buffer/' + label_name
      np.save(buffer_path_input, memory_buffer_inputs)
      np.save(buffer_path_label, memory_buffer_labels)
      
      # print ("Task 1 finished. Testing acc")
      # acc = training_environment.validate(model, mode='testing', batch_size=-1, statistics=False)

    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))



    



# Perform ODDA
# odda = False
# if odda:
#   # Load pretrained model
#   model.load_state_dict(torch.load('./pretrain2_ordered_v3_fixFilfixUttr.pth', map_location=torch.device('cuda')))
#   environmental_noise = 'TMETRO'
#   training_environment.adapt(model, environmental_noise)

# # Accuracy on the training set. 
# print ("Training acc")
# acc = training_environment.validate(model, mode='training', batch_size=-1, statistics=False)
# Accuracy on the validation set. 
# print ("Validation acc")
# acc = training_environment.validate(model, mode='validation', batch_size=-1, statistics=False)
# Accuracy on the testing set. 
print ("Testing acc")
acc = training_environment.validate(model, mode='testing', batch_size=-1, statistics=False)

