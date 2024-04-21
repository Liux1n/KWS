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
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch),
#         Liuxin Qing, ETH(liuqing@student.ethz.ch)

import copy
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
from utils import remove_txt, parameter_generation, load_config, task_average_forgetting, Buffer_CB,  Buffer_NRS, Buffer_CB_fast
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

parser.add_argument('--mode', 
                  choices=['cil', 'dil'], 
                  default='cil',
                  required=True,
                  help='training mode (default: cil)')

parser.add_argument('--method', 
                  choices=['source_only', 'finetune', 'joint', 'ER_NRS', 'ECBRS', 'ER_CB', 'ER_CB_fast',
                           'base_pretrain', 'joint_pretrain', 'final_test'
                           ], 
                  default='base_pretrain',
                  required=True,
                  help='training  method (default: base)')

parser.add_argument('--noise_mode', choices=['nlkws', 'nakws', 'odda'], default='nakws',
                    help='noise mode')
parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
parser.add_argument('--wandb', action='store_true', help="Enable wandb log.")
parser.add_argument('--pretrain', action='store_true', help="Enable pretrain.")
parser.add_argument('--background_volume', type=int, help="Set background volume.")
parser.add_argument('--early_stop', action='store_true', help="Enable wandb log.")
parser.add_argument('--forgetting', action='store_true', help="Enable wandb log.")
args = parser.parse_args()

time_str = time.strftime("%Y%m%d-%H%M%S")
run_name = args.mode + '_' + args.method + '_' + time_str
if args.wandb:
  wandb.init(
            project=config['project_name'], 
            entity="liux1n", 
            name=run_name,
            config=config
            )


if args.debug:
  print('Debug mode, 1 epoch.')

# Load pretrained model 
# DIL models:
# model_name = 'base_dil_task_0_model.pth'
# model_name = 'base_dil_joint_model.pth'
# model_name = 'base_12_dil_task_0_model.pth'

# DIL joint pretrain
# model_name = 'dil_joint_pretrain_20240419-03430430.pth'

# DIL base pretrain
# model_name = 'dil_base_pretrain_vol_5.pth'
# model_name = 'dil_base_pretrain_vol_10.pth'
# model_name = 'dil_joint_pretrain_vol_10.pth'

# CIL joint
# model_name = 'cil_joint_pretrain_20240418-23171014.pth'

# CIL base pretrain
model_name = 'cil_base_pretrain_20240419-02554315.pth'

# CIL er_nrs


# model_name = 'joint_12_cil_joint_model.pth'

model_path = './models/' + model_name 


if args.mode == 'dil':

  if args.method == 'base_pretrain':
    # if args.background_volume:
    #   config['background_volume'] = args.background_volume
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_task_0')

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    print('Training model on dil_task_0...')
    n_classes = config['n_classes'] + 2
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    
    start=time.process_time()
    
    training_environment.train(model,task_id=None)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
  

  elif args.method == 'joint_pretrain':
    # if args.background_volume:
    #   config['background_volume'] = args.background_volume

    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_joint')

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    print('Joint-Training model...')
    n_classes = config['n_classes'] + 2
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    
    start=time.process_time()
    training_environment.train(model,task_id=None)

    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
  

  elif args.method == 'joint':
    print('Using joint-training model. No continual training needed.')
    # tasks = ['dil_task_0', 'dil_task_1', 'dil_task_2', 'dil_task_3']
    # task_id = ['dil_task_0','dil_task_1_disjoint', 'dil_task_2_disjoint', 'dil_task_3_disjoint']
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    
    # acc_matrix[0,0] = acc after training task 0 on task 0
    # acc_matrix[0,1] = acc after training task 0 on task 1
    # acc_matrix[0,2] = acc after training task 0 on task 2
    # acc_matrix[0,3] = acc after training task 0 on task 3

    # acc_matrix[1,0] = acc after training task 1 on task 0
    # acc_matrix[1,1] = acc after training task 1 on task 1
    # acc_matrix[1,2] = acc after training task 1 on task 2
    # acc_matrix[1,3] = acc after training task 1 on task 3

    # acc_matrix[2,0] = acc after training task 2 on task 0
    # acc_matrix[2,1] = acc after training task 2 on task 1
    # acc_matrix[2,2] = acc after training task 2 on task 2
    # acc_matrix[2,3] = acc after training task 2 on task 3

    # acc_matrix[3,0] = acc after training task 3 on task 0
    # acc_matrix[3,1] = acc after training task 3 on task 1
    # acc_matrix[3,2] = acc after training task 3 on task 2
    # acc_matrix[3,3] = acc after training task 3 on task 3
    # final acc: acc_matrix[3,:]
    acc_done = 0
    for i in range(4): # i: 0, 1, 2, 3
        
      task_id = f'dil_task_{i}'
      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)

      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      # print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))
      
      # Loaded model has n_classes = 35 + 2 = 37
      n_classes = config['n_classes'] + 2 # 35 + 2
      model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
      model.to(device)
      
      dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
      # count_ops(model, dummy_input)
      model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

      training_environment = Train(audio_processor, training_parameters, model, device, args, config)

      
      # Training
      print ("Testing Accuracy on ", task_id, '...')
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
      print(f'Test Accuracy of Task {i}: ', acc_task)
      if args.wandb:
          wandb.log({f'ACC_task_{i}': acc_task})
      
      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        task_id_disjoint = ['dil_task_0_disjoint','dil_task_1_disjoint', 'dil_task_2_disjoint', 'dil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id_disjoint[j])
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')

    print('acc_matrix:', acc_matrix)

    average_forgetting = task_average_forgetting(acc_matrix)

    print("Task-average Forgetting:", average_forgetting)
    if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})

    # acc_matrix: [[70.56831323 66.97505369 69.06492648 66.80158599]
    #              [70.56831323 66.97505369 69.06492648 66.80158599]
    #              [70.56831323 66.97505369 69.06492648 66.80158599]
    #              [70.56831323 66.97505369 69.06492648 66.80158599]]
    

  elif args.method == 'source_only':
    print('Using source-only model. No continual training needed.')
    # tasks = ['dil_task_0', 'dil_task_1', 'dil_task_2', 'dil_task_3']
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    for i in range(4):
        
      task_id = f'dil_task_{i}'
      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)

      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))
      
      # Loaded model has n_classes = 35 + 2 = 37
      n_classes = config['n_classes'] + 2 # 35 + 2
      model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
      model.to(device)
      
      dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
      # count_ops(model, dummy_input)
      model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
      if i == 0:
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        print('Model loaded from ', model_path)
      
      training_environment = Train(audio_processor, training_parameters, model, device, args, config)
      print ("Testing Accuracy on ", task_id, '...')
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
      print(f'Test Accuracy of Task {i}: ', acc_task)
      if args.wandb:
          wandb.log({f'ACC_task_{i}': acc_task})

      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        task_id_disjoint = ['dil_task_0_disjoint','dil_task_1_disjoint', 'dil_task_2_disjoint', 'dil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id_disjoint[j])
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
      else:
        del audio_processor
        del training_environment

    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
          wandb.log({'Task-average Forgetting': average_forgetting})

  
  elif args.method == 'finetune':
     
    print('Start Fine-tuning...')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})

      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Fine-tuning on {task_id}...')
        model = training_environment.finetune(model, task_id=task_id)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})

      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        del audio_processor
        del training_environment
        task_id_disjoint = ['dil_task_0_disjoint','dil_task_1_disjoint', 'dil_task_2_disjoint', 'dil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id_disjoint[j])
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
      else:
        del audio_processor
        del training_environment

    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})

    # Save the model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.mode + '_' + args.method + '_' + timestr + '.pth'
    PATH = './models/' + model_name
    torch.save(model.state_dict(), PATH)
    print('Model saved at ', PATH)
  

  elif args.method == 'ER_NRS':
     
    print('Start ER_NRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer
        memory_buffer = Buffer_NRS(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            memory_buffer.add_data(inputs_mb, labels_mb)
        print('Memory buffer initialized. Size:', memory_buffer.get_size())
        # delete data
        del data
        
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})

      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        # # reset num_seen_examples in memory buffer
        # memory_buffer.reset_num_seen_examples()
        model, memory_buffer = training_environment.ER_NRS(model, memory_buffer, task_id)
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
      
      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        del audio_processor
        del training_environment
        task_id_disjoint = ['dil_task_0_disjoint','dil_task_1_disjoint', 'dil_task_2_disjoint', 'dil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id_disjoint[j])
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
      else:
        del audio_processor
        del training_environment

    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})

    # Save the model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.mode + '_' + args.method + '_' + timestr + '.pth'
    PATH = './models/' + model_name
    torch.save(model.state_dict(), PATH)
    print('Model saved at ', PATH)
        

elif args.mode == 'cil':

  if args.method == 'base_pretrain':
    print('Training model on cil_task_0...')
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    n_classes = 19
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    # print('Task '+args.task)
    # Task 0. Train from scratch
    # start=time.clock_gettime(0)
    start=time.process_time()
    
    training_environment.train(model,task_id='cil_task_0')
    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
    acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
    print(f'Test Accuracy: ', acc_task)


  elif args.method == 'joint_pretrain':
    print('Joint-Training model...')
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    n_classes = 37
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    # print('Task '+args.task)
    # Task 0. Train from scratch
    # start=time.clock_gettime(0)
    start=time.process_time()
    
    training_environment.train(model,task_id='cil_joint')
    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))


  elif args.method == 'joint':
    print('Using joint-training model. No continual training needed.')
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    # Loaded model has n_classes = 35 + 2 = 37
    n_classes = config['n_classes'] + 2 # 35 + 2
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    print('Using joint-training model. No continual training needed.')
    print ("Testing Accuracy...")
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    for i in range(4):
      task_id = f'cil_task_{i}'
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
      print(f'Test Accuracy of Task {i}: ', acc_task)
      if args.wandb:
          wandb.log({f'ACC_task_{i}': acc_task})

      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        task_id_disjoint = ['cil_task_0_disjoint','cil_task_1_disjoint', 'cil_task_2_disjoint', 'cil_task_3_disjoint']
        for j in range(i+1): 
          # training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id_disjoint[j])
          # # Dataset generation
          # audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          # training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id_disjoint[j], statistics=False)
          acc_matrix[i,j] = acc_task
          # del audio_processor
          # del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
    
    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})


  elif args.method == 'source_only':
    # Loaded model has n_classes = 19
    print('Using source-only model. No continual training needed.')
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    for i in range(4):
      n_classes = 19
      model = DSCNNS(use_bias = True, n_classes = n_classes)
      model.to(device)
      
      dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
      # count_ops(model, dummy_input)
      model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
      task_id = str(f'cil_task_{i}')
      if i == 0:
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
      else:
        new_classes = 6 * i
        fc_new= torch.nn.Linear(64, n_classes + new_classes)
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

      training_environment = Train(audio_processor, training_parameters, model, device, args, config)
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
      print(f'Test Accuracy of Task {i}: ', acc_task)
      if args.wandb:
          wandb.log({f'ACC_task_{i}': acc_task})
      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        task_id_disjoint = ['cil_task_0_disjoint','cil_task_1_disjoint', 'cil_task_2_disjoint', 'cil_task_3_disjoint']
        for j in range(i+1):
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id_disjoint[j], statistics=False)
          acc_matrix[i,j] = acc_task
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})
        

  elif args.method == 'finetune':
     
    print('Start Fine-tuning...')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    n_classes = 19
    new_classes_per_task = 6
    
    for i, task_id in enumerate(tasks): # i: 0, 1, 2
      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      # Removing stored inputs and activations
      remove_txt()
      if i == 0:
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        # print(f'Fine-tuning on {task_id}...')
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
      else:
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
            weight = copy.deepcopy(model.fc1.weight.data)
            bias = copy.deepcopy(model.fc1.bias.data)
            # copy old weights and biases to fc_new
            fc_new.weight.data[:n_classes] = weight
            fc_new.bias.data[:n_classes] = bias
            # replace fc1 with fc_new
            model.fc1 = fc_new

        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Fine-tuning on {task_id}...')
        model = training_environment.finetune(model, task_id)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task

      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        del audio_processor
        del training_environment
        task_id_disjoint = ['cil_task_0_disjoint','cil_task_1_disjoint', 'cil_task_2_disjoint', 'cil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id_disjoint[j], statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
      else:
        del audio_processor
        del training_environment
    
    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})
      
    # Save the model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.mode + '_' + args.method + '_' + timestr + '.pth'
    PATH = './models/' + model_name
    torch.save(model.state_dict(), PATH)
    print('Model saved at ', PATH)
        

  elif args.method == 'ER_NRS':

    print('Start ER_NRS')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    n_classes = 19
    new_classes_per_task = 6
    
    for i, task_id in enumerate(tasks): # i: 0, 1, 2
      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      # Removing stored inputs and activations
      remove_txt()
      if i == 0:
        # initialize memory buffer
        memory_buffer = Buffer_NRS(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, 'cil_task_0', task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            memory_buffer.add_data(inputs_mb, labels_mb)
        print('Memory buffer initialized. Size:', memory_buffer.get_size())
        # delete data
        del data
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
      else:
        
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
            weight = copy.deepcopy(model.fc1.weight.data)
            bias = copy.deepcopy(model.fc1.bias.data)
            # copy old weights and biases to fc_new
            fc_new.weight.data[:n_classes] = weight
            fc_new.bias.data[:n_classes] = bias
            # replace fc1 with fc_new
            model.fc1 = fc_new

        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')

        model, memory_buffer = training_environment.ER_NRS(model, memory_buffer, task_id)
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
      
      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        del audio_processor
        del training_environment
        task_id_disjoint = ['cil_task_0_disjoint','cil_task_1_disjoint', 'cil_task_2_disjoint', 'cil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id_disjoint[j], statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
      else:
        del audio_processor
        del training_environment
    
    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})

    # Save the model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.mode + '_' + args.method + '_' + timestr + '.pth'
    PATH = './models/' + model_name
    torch.save(model.state_dict(), PATH)
    print('Model saved at ', PATH)


  elif args.method == 'ER_CB_fast':

    print('Start ER_NRS')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    n_classes = 19
    new_classes_per_task = 6
    
    for i, task_id in enumerate(tasks): # i: 0, 1, 2
      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      # Removing stored inputs and activations
      remove_txt()
      if i == 0:
        # initialize memory buffer
        memory_buffer = Buffer_CB_fast(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, 'cil_task_0', task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            memory_buffer.add_data(inputs_mb, labels_mb)
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
        # delete data
        del data
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
      else:
        
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
            weight = copy.deepcopy(model.fc1.weight.data)
            bias = copy.deepcopy(model.fc1.bias.data)
            # copy old weights and biases to fc_new
            fc_new.weight.data[:n_classes] = weight
            fc_new.bias.data[:n_classes] = bias
            # replace fc1 with fc_new
            model.fc1 = fc_new

        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')

        model, memory_buffer = training_environment.ER_NRS(model, memory_buffer, task_id)
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
      
      if args.forgetting:
        print('Testing on Disjoint Tasks...')
        del audio_processor
        del training_environment
        task_id_disjoint = ['cil_task_0_disjoint','cil_task_1_disjoint', 'cil_task_2_disjoint', 'cil_task_3_disjoint']
        for j in range(i+1): 
          training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)
          # Dataset generation
          audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
          training_environment = Train(audio_processor, training_parameters, model, device, args, config)
          # print (f"Testing Accuracy on {task_id_disjoint}...")
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id_disjoint[j], statistics=False)
          acc_matrix[i,j] = acc_task
          del audio_processor
          del training_environment
          acc_done += 1
          print(f'Finished Testing for Acc Matrix {acc_done}/10')
      else:
        del audio_processor
        del training_environment
    
    if args.forgetting:
      print('acc_matrix:', acc_matrix)

      average_forgetting = task_average_forgetting(acc_matrix)

      print("Task-average Forgetting:", average_forgetting)
      if args.wandb:
        wandb.log({'Task-average Forgetting': average_forgetting})

    # Save the model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.mode + '_' + args.method + '_' + timestr + '.pth'
    PATH = './models/' + model_name
    torch.save(model.state_dict(), PATH)
    print('Model saved at ', PATH)



  elif args.method == 'final_test':
    print('Using final_test model. No continual training needed.')
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    # Loaded model has n_classes = 35 + 2 = 37
    n_classes = config['n_classes'] + 2 # 35 + 2
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)

    print('Testing on Disjoint Tasks...')
    # task_id_disjoint = ['cil_task_0_disjoint','cil_task_1_disjoint', 'cil_task_2_disjoint', 'cil_task_3_disjoint']
    task_id_disjoint = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    for i in range(4): 
      # print('Testing on ', task_id_disjoint[i])
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id_disjoint[i], statistics=False)
      print(f'Test Accuracy of {task_id_disjoint[i]}: ', acc_task)
    # if args.forgetting:
    #   print('acc_matrix:', acc_matrix)

    #   average_forgetting = task_average_forgetting(acc_matrix)

    #   print("Task-average Forgetting:", average_forgetting)
    #   if args.wandb:
    #     wandb.log({'Task-average Forgetting': average_forgetting})
        


