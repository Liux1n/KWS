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
import dataset_copy
import os
import time
import math
import random
import wandb  
import numpy as np

from torchsummary import summary
# from model import DSCNNS, DSCNNM, DSCNNL
from dscnn import DSCNNS, DSCNNM, DSCNNL
from utils import remove_txt, parameter_generation, load_config, task_average_forgetting, \
                  Buffer_NRS, Buffer_ECB, Buffer_DAECB, Buffer_LAECB, Buffer_LDAECB, Buffer_DAECB_entropy, \
                  Buffer_AAECB, Buffer_GAECB, Buffer_LDAAECB, Buffer_DAECB_reg
from copy import deepcopy
from pthflops import count_ops
from train import Train
import argparse

# from methods import base_pretrain, joint_pretrain


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.cuda.set_device(1)

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print ('Torch Version:', torch.version.__version__)
print('Device:', device)
config = load_config("config.yaml")
seed = config['seed']
torch.manual_seed(seed)
parser = argparse.ArgumentParser()

parser.add_argument('--mode', 
                  choices=['cil', 'dil'], 
                  default='cil',
                  required=True,
                  help='training mode (default: cil)')

parser.add_argument('--method', 
                  choices=['source_only', 'finetune', 'joint', 'NRS', 'ECBRS', 'DAECBRS', 'LAECBRS', 'ClusterECBRS',
                           'base_pretrain', 'joint_pretrain', 'custom', 'LossECB', 'noise_data_test', 'LDAECBRS', 'feat_vis',
                           'DAECBRS_entropy', 'AAECBRS', 'feat_vis_noise', 'GAECBRS', 'LDAAECBRS', 'DAECBRS_reg', 'feat_vis_noise_data_test', 
                           'noise_data_train', 'noise_data'
                           ], 
                  default='base_pretrain',
                  required=True,
                  help='training  method (default: base)')

parser.add_argument('--noise_mode', choices=['nlkws', 'nakws'], default='nakws',
                    required=True,
                    help='noise mode')

parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
parser.add_argument('--wandb', action='store_true', help="Enable wandb log.")
parser.add_argument('--background_volume', type=int, help="Set background volume.")
parser.add_argument('--early_stop', action='store_true', help="Enable early_stop.")
parser.add_argument('--forgetting', action='store_true', help="Enable forgetting measure.")
parser.add_argument('--dynamic_loss', action='store_true', help="Enable dynamic_loss.")
parser.add_argument('--augmentation', action='store_true', help="Enable data augmentation.")
parser.add_argument('--remark', type=str,default='_', help='Remark of the run.')
parser.add_argument('--epoch', type=int, help='Number of epochs for training.')
parser.add_argument('--patience', type=int, help='Number of epochs for training.')
parser.add_argument('--epoch_cl', type=int, help='Number of epochs for continual training.')
parser.add_argument('--n_clusters', type=int, help='Number of clusters in ClusterECBRS.')
parser.add_argument('--buffer_size', type=int, help='Number of clusters in ClusterECBRS.')
parser.add_argument('--load_buffer', action='store_true', help="Enable load_buffer.")
parser.add_argument('--load_model', action='store_true', help="Enable load_model.")
parser.add_argument('--from_scratch', action='store_true', help="Enable load_model.")
parser.add_argument('--kd', action='store_true', help="Enable knowledge distillation.")
parser.add_argument('--mixup', action='store_true', help="Enable knowledge distillation.")
parser.add_argument('--reg', action='store_true', help="Enable regularizer.")
parser.add_argument('--task1_joint', action='store_true', help="Enable regularizer.")
parser.add_argument('--task2_joint', action='store_true', help="Enable regularizer.")
parser.add_argument('--model_M', action='store_true', help="Enable regularizer.")
parser.add_argument('--model_L', action='store_true', help="Enable regularizer.")
parser.add_argument('--save_buffer', action='store_true', help="Enable load_model.")
parser.add_argument('--seed', type=int, help='Number of epochs for training.')
parser.add_argument('--snr', type=int, help='Number of epochs for training.')
args = parser.parse_args()


time_str = time.strftime("%Y%m%d-%H%M%S")
if args.epoch:
  config['epochs'] = args.epoch
  print('Number of epochs changed to ', args.epoch)

if args.buffer_size:
  config['memory_buffer_size'] = args.buffer_size
  print('Buffer size changed to ', args.buffer_size)

if args.seed:
  config['seed'] = args.seed
  torch.manual_seed(args.seed)
  print('Seed changed to ', args.seed)

if args.n_clusters:
  config['n_clusters'] = args.n_clusters
  print('Number of clusters changed to ', args.n_clusters)

if args.epoch_cl:
  config['epochs_cl'] = args.epoch_cl
  print('Number of epochs_cl changed to ', args.epoch_cl)

if args.snr:
  config['snr'] = args.snr
  print('SNR changed to ', args.snr)

if args.patience:
  config['patience'] = args.patience
  print('Patience changed to ', args.patience)

if args.remark is None:
  run_name = args.mode + '_' + args.method + '_' + time_str
else:
  run_name = args.mode + '_' + args.method + '_' + args.remark + '_' + time_str

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


# model_name = 'cil_base_pretrain_new.pth' # CIL base
# model_name = 'cil_joint_pretrain_new.pth' # CIL joint
# model_name = 'cil_LAECBRS_20240502-06335714_base.pth'
# model_name = 'cil_joint_pretrain_20240526-020927_patience20.pth' # joint patience 20



# model_name = 'cil_LAECBRS_20240503-14264014_base_seed4.pth' # seed4 LAECBRS bases
# model_name = 'cil_LAECBRS_20240504-10112315_base_seed5.pth' # seed 5
# model_name = 'cil_LAECBRS_20240504-10071015_base_seed42.pth' # seed 42

model_name = 'cil_base_pretrain_20240611-234943_final_modelM.pth'




# model_name = 'cil_NRS_20240506-133410.pth' # seed 42 NRS finished
# model_name = 'cil_finetune_20240506-102537.pth' # seed 42 finetune finished
# model_name = 'cil_LAECBRS_20240504-164346.pth' # seed 42 LAECBRS finished
# model_name = 'cil_ECBRS_20240504-182311.pth' # seed 42 ECBRS finished
# model_name = 'cil_DAECBRS_20240504-185845.pth' # seed 42 DAECBRS finished
# model_name = 'cil_DAECBRS_entropy_20240510-203842.pth' # seed 42 DAECBRS_entropy finished


# model_name = 'dil_base_pretrain_20240515-002545_noise_fixed_seed42_snr-8.pth'
# model_name = 'dil_ECBRS_20240519-191914.pth' # seed98 ECBRS finished

# model_name = 'dil_noise_data_test_20240531-152538_seed4_epoch2.pth'
# model_name = 'dil_noise_data_test_20240530-183842_seed4_patience5.pth' # seed4 noise_data_test finished
# model_name = 'dil_noise_data_test_20240530-210615_seed4_epoch5.pth' # seed4 noise_data_test 5epoch finished
# model_name = 'dil_base_pretrain_20240531-014143_trial1.pth'
# model_name = 'dil_base_pretrain_20240531-084058_trial3_newDivision.pth'
# model_name = 'dil_base_pretrain_20240531-083410_trial3.pth' # trial3
# model_name = 'dil_base_pretrain_20240531-222531_trial4.pth' # trial4
# model_name = 'dil_base_pretrain_20240531-191933_trial5.pth' # trial5
# model_name = 'dil_base_pretrain_20240531-222531_trial4.pth' # trial6
# model_name = 'dil_base_pretrain_20240531-231423_trial7.pth' # trial7
# model_name = 'dil_base_pretrain_20240601-084512_trial8.pth' # trial8
# model_name = 'dil_base_pretrain_20240601-210931_trial9.pth' # trial9
# model_name = 'dil_base_pretrain_20240601-220630_trial10.pth' # trial10
# model_name = 'dil_base_pretrain_20240604-022944_trial10.pth'
# model_name = 'dil_base_pretrain_20240605-013117_trial12.pth' # trial12
# model_name = 'dil_base_pretrain_20240605-060514_trial13.pth' # trial13
# model_name = 'dil_base_pretrain_20240606-030301_trial17_KMedoids.pth' # trial17 KMedoids
# model_name = 'dil_base_pretrain_20240606-044521_trial19.pth'
# model_name = 'dil_base_pretrain_20240608-230512_final.pth' # final base

# model_name = 'dil_joint_pretrain_20240609-012816_final.pth' # final joint




# saved buffer state
buffer_name = 'buffer_cil_LAECBRS_20240504-100710.pth' # seed 42



model_path = './models/' + model_name 




if args.mode == 'dil':

  
  if args.method == 'base_pretrain':
    # if args.background_volume:
    #   config['background_volume'] = args.background_volume
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_task_0')

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
    audio_processor.divide_data_set(training_parameters, task_id='dil_task_0')
    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    print('Training model on dil_task_0...')
    n_classes = config['n_classes'] + 2
    if args.model_M:
      model = DSCNNM(use_bias = True, n_classes = n_classes)
    elif args.model_L:
      model = DSCNNL(use_bias = True, n_classes = n_classes)
    else:
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
    if args.task1_joint:
      task_id = 'dil_task_1_joint'
    elif args.task2_joint:
      task_id = 'dil_task_2_joint'
    else:
      task_id = 'dil_joint'
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)

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
    if args.model_M:
      model = DSCNNM(use_bias = True, n_classes = n_classes)
    elif args.model_L:
      model = DSCNNL(use_bias = True, n_classes = n_classes)
    else:
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
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j

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
      if args.model_M:
        model = DSCNNM(use_bias = True, n_classes = n_classes)
      elif args.model_L:
        model = DSCNNL(use_bias = True, n_classes = n_classes)
      else:
        model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
      model.to(device)
      
      dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
      # count_ops(model, dummy_input)
      model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

      training_environment = Train(audio_processor, training_parameters, model, device, args, config)

      
      # Training
      print ("Testing Accuracy on ", task_id, '...')
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=True)
      # acc_task = training_environment.validate_test(model, mode='validation', batch_size=1000, task_id=task_id, statistics=True)

      print(f'Test Accuracy of Task {i}: ', acc_task)
      if args.wandb:
          wandb.log({f'ACC_task_{i}': acc_task})
      
    
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
      audio_processor.divide_data_set(training_parameters, task_id=task_id)

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
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
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
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
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
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
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
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})

      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Fine-tuning on {task_id}...')
        model = training_environment.finetune(model, task_id=task_id)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
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
          acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
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
  

  elif args.method == 'NRS':
     
    print('Start NRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
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
            # inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            # labels_mb = torch.Tensor(labels_mb).long().to(device)
            inputs_mb, labels_mb, _ = data[0]
            inputs_mb.unsqueeze_(1)
            memory_buffer.add_data(inputs_mb, labels_mb)
            if minibatch % 50 == 0:
                cls_count = memory_buffer.get_class_count()
                print("Class count: ", cls_count)
                # print("Time for adding data: ", (end_time - start_time)/config['batch_size']*1000, 'ms per sample')
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
        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        
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
        
  
  elif args.method == 'ECBRS':
     
    print('Start ECBRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer
        memory_buffer = Buffer_ECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            memory_buffer.add_data(inputs_mb, labels_mb)
            if minibatch % 20 == 0:
              print(f'{minibatch}/{int(len(data))}')
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
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
        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        
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

  
  elif args.method == 'DAECBRS':
     
    print('Start DAECBRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer
        memory_buffer = Buffer_DAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            memory_buffer.add_data(inputs_mb, labels_mb)
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
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
        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        
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
  
    
  elif args.method == 'DAECBRS_entropy':
     
    print('Start DAECBRS_entropy')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer
        memory_buffer = Buffer_DAECB_entropy(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            memory_buffer.add_data(inputs_mb, labels_mb)
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
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
        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        
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
  

  elif args.method == 'LAECBRS':
     
    print('Start LAECBRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer

        memory_buffer = Buffer_LAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        
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
        
        # start=time.process_time()
        
        # training_environment.train(model,task_id=None)
        # print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
        start_time_training = time.time()
        if args.load_buffer:
          # model_ckpt = 'cil_LAECBRS_20240503-14264014_acc87.0998475609756.pth' # seed 4
          model_ckpt = 'cil_LAECBRS_20240504-10071015_base_seed42.pth' # seed 42
          model_ckpt_path = './models/' + model_ckpt
          model.load_state_dict(torch.load(model_ckpt_path, map_location=torch.device('cuda')))
          # buffer_name = 'buffer_cil_LAECBRS_20240503-142640.pth' # seed 4
          buffer_name = 'buffer_cil_LAECBRS_20240504-100710.pth' # seed 42
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        else:

          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          # model_path, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
          # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          if args.save_buffer:
            num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

            # make them a big dictionary
            buffer_state = {'num_seen_examples': num_seen_examples, 
                            'buffer': buffer, 
                            'loss': loss, 
                            'loss_index': loss_index, 
                            'class_count': class_count, 
                            'class_count_total': class_count_total, 
                            'full_classes': full_classes}
            # save buffer_state
            timestr = time.strftime("%Y%m%d-%H%M%S")
            buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
            buffer_path = './buffer_state/' + buffer_name
            torch.save(buffer_state, buffer_path)
            print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))

        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})


      # task 1, 2, 3
      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        # # reset num_seen_examples in memory buffer
        # memory_buffer.reset_num_seen_examples()
        model, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
        
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
  

  elif args.method == 'LDAECBRS':
     
    print('Start LDAECBRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer

        memory_buffer = Buffer_LDAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        
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
        
        # start=time.process_time()
        
        # training_environment.train(model,task_id=None)
        # print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
        start_time_training = time.time()
        if args.load_buffer:
          # model_ckpt = 'cil_LAECBRS_20240503-14264014_acc87.0998475609756.pth' # seed 4
          model_ckpt = 'cil_LAECBRS_20240504-10071015_base_seed42.pth' # seed 42
          model_ckpt_path = './models/' + model_ckpt
          model.load_state_dict(torch.load(model_ckpt_path, map_location=torch.device('cuda')))
          # buffer_name = 'buffer_cil_LAECBRS_20240503-142640.pth' # seed 4
          buffer_name = 'buffer_cil_LAECBRS_20240504-100710.pth' # seed 42
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        else:

          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_LDAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_LDAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          # model_path, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
          # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          if args.save_buffer:
            num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

            # make them a big dictionary
            buffer_state = {'num_seen_examples': num_seen_examples, 
                            'buffer': buffer, 
                            'loss': loss, 
                            'loss_index': loss_index, 
                            'class_count': class_count, 
                            'class_count_total': class_count_total, 
                            'full_classes': full_classes}
            # save buffer_state
            timestr = time.strftime("%Y%m%d-%H%M%S")
            buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
            buffer_path = './buffer_state/' + buffer_name
            torch.save(buffer_state, buffer_path)
            print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))

        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})


      # task 1, 2, 3
      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        # # reset num_seen_examples in memory buffer
        # memory_buffer.reset_num_seen_examples()
        model, memory_buffer = training_environment.ER_LDAECB(model, memory_buffer, task_id)
        
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
  
  
  elif args.method == 'GAECBRS':
     
    print('Start GAECBRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer

        memory_buffer = Buffer_GAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        
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
        
        # start=time.process_time()
        
        # training_environment.train(model,task_id=None)
        # print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
        start_time_training = time.time()
        if args.load_buffer:
          # model_ckpt = 'cil_LAECBRS_20240503-14264014_acc87.0998475609756.pth' # seed 4
          model_ckpt = 'cil_LAECBRS_20240504-10071015_base_seed42.pth' # seed 42
          model_ckpt_path = './models/' + model_ckpt
          model.load_state_dict(torch.load(model_ckpt_path, map_location=torch.device('cuda')))
          # buffer_name = 'buffer_cil_LAECBRS_20240503-142640.pth' # seed 4
          buffer_name = 'buffer_cil_LAECBRS_20240504-100710.pth' # seed 42
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        else:

          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_GAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_GAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          # model_path, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
          # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          if args.save_buffer:
            num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

            # make them a big dictionary
            buffer_state = {'num_seen_examples': num_seen_examples, 
                            'buffer': buffer, 
                            'loss': loss, 
                            'loss_index': loss_index, 
                            'class_count': class_count, 
                            'class_count_total': class_count_total, 
                            'full_classes': full_classes}
            # save buffer_state
            timestr = time.strftime("%Y%m%d-%H%M%S")
            buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
            buffer_path = './buffer_state/' + buffer_name
            torch.save(buffer_state, buffer_path)
            print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))

        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})


      # task 1, 2, 3
      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        # # reset num_seen_examples in memory buffer
        # memory_buffer.reset_num_seen_examples()
        model, memory_buffer = training_environment.ER_GAECB(model, memory_buffer, task_id)
        
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
  

  elif args.method == 'AAECBRS':
     
    print('Start AAECBRS')
    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = config['n_classes'] + 2 # 35 + 2
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0
    for i, task_id in enumerate(tasks): # i: 0, 1, 2, 3

      training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)
      # Dataset generation
      audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
      audio_processor.divide_data_set(training_parameters, task_id=task_id)
      train_size = audio_processor.get_size('training')
      valid_size = audio_processor.get_size('validation')
      test_size = audio_processor.get_size('testing')
      print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

      if i == 0:
        # initialize memory buffer

        memory_buffer = Buffer_AAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, task_id, task = None)
        
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
        
        # start=time.process_time()
        
        # training_environment.train(model,task_id=None)
        # print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))
        start_time_training = time.time()
        if args.load_buffer:
          # model_ckpt = 'cil_LAECBRS_20240503-14264014_acc87.0998475609756.pth' # seed 4
          model_ckpt = 'cil_LAECBRS_20240504-10071015_base_seed42.pth' # seed 42
          model_ckpt_path = './models/' + model_ckpt
          model.load_state_dict(torch.load(model_ckpt_path, map_location=torch.device('cuda')))
          # buffer_name = 'buffer_cil_LAECBRS_20240503-142640.pth' # seed 4
          buffer_name = 'buffer_cil_LAECBRS_20240504-100710.pth' # seed 42
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        else:

          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_AAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_AAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

          # make them a big dictionary
          buffer_state = {'num_seen_examples': num_seen_examples, 
                          'buffer': buffer, 
                          'loss': loss, 
                          'loss_index': loss_index, 
                          'class_count': class_count, 
                          'class_count_total': class_count_total, 
                          'full_classes': full_classes}
          # save buffer_state
          timestr = time.strftime("%Y%m%d-%H%M%S")
          buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
          buffer_path = './buffer_state/' + buffer_name
          torch.save(buffer_state, buffer_path)
          print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))

        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=None, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})


      # task 1, 2, 3
      else:
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        # # reset num_seen_examples in memory buffer
        # memory_buffer.reset_num_seen_examples()
        model, memory_buffer = training_environment.ER_AAECB(model, memory_buffer, task_id)
        
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
  

  elif args.method == 'feat_vis':

    print('Start feature visualization')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    n_classes = 37
    new_classes_per_task = 6
    task_id = 'cil_task_3'
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)  # To be parametrized
    training_parameters['batch_size'] = 2048
    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)

    start=time.process_time()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    print('model loaded from ', model_path)
    training_environment.feat_vis(model, task_id=task_id, statistics=False)


  elif args.method == 'feat_vis_noise':

    print('Start feature visualization')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
    n_classes = 37
    new_classes_per_task = 6
    task_id = 'dil_task_3'
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)  # To be parametrized
    training_parameters['batch_size'] = 4096
    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)

    start=time.process_time()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    print('model loaded from ', model_path)
    training_environment.feat_vis_noise(model, model_name, task_id=task_id, statistics=False)


  # elif args.method == 'noise_data_test':

  #   print('Start noise data test')

  #   acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
  #   acc_done = 0

  #   tasks = ['dil_task_0','dil_task_1', 'dil_task_2', 'dil_task_3']
  #   n_classes = 37
  #   new_classes_per_task = 6
  #   task_id = 'dil_task_3'
  #   training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)  # To be parametrized
  #   training_parameters['batch_size'] = 4096
  #   # Dataset generation
  #   audio_processor = dataset_copy.AudioProcessor(training_parameters, data_processing_parameters)

  #   train_size = audio_processor.get_size('training')
  #   valid_size = audio_processor.get_size('validation')
  #   test_size = audio_processor.get_size('testing')
  #   print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

  #   model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
  #   model.to(device)
  #   summary(model,(1,49,data_processing_parameters['feature_bin_count']))
  #   dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
  #   # count_ops(model, dummy_input)

  #   # Training initialization
  #   training_environment = Train(audio_processor, training_parameters, model, device, args, config)

  #   start=time.process_time()

  #   model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
  #   print('model loaded from ', model_path)
  #   training_environment.noise_data_test(model, model_name, task_id=task_id, statistics=False)

    
  elif args.method == 'noise_data_train':
    # if args.background_volume:
    #   config['background_volume'] = args.background_volume
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_joint')

    # Dataset generation
    audio_processor = dataset_copy.AudioProcessor(training_parameters, data_processing_parameters)
    # audio_processor.divide_data_set(training_parameters, task_id='dil_task_0')
    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    print('Training model on dil_task_0...')
    # n_classes = config['n_classes'] + 2
    n_classes = 18
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 18 noise classes
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    
    start=time.process_time()
    
    training_environment.noise_data_test(model,task_id=None)
    training_environment.validate_noise(model, mode='testing', batch_size=-1, task_id=None)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))

  elif args.method == 'noise_data_test':
    # if args.background_volume:
    #   config['background_volume'] = args.background_volume
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_joint')

    # Dataset generation
    audio_processor = dataset_copy.AudioProcessor(training_parameters, data_processing_parameters)
    # audio_processor.divide_data_set(training_parameters, task_id='dil_task_0')
    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    # Removing stored inputs and activations
    remove_txt()
    # n_classes = config['n_classes'] + 2
    n_classes = 18
    model = DSCNNS(use_bias = True, n_classes = n_classes) # 18 noise classes
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)
    
    start=time.process_time()
    
    training_environment.validate_noise(model, mode='testing', batch_size=-1, task_id=None)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))

  elif args.method == 'noise_data':
    # if args.background_volume:
    #   config['background_volume'] = args.background_volume
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_joint')
    training_parameters['batch_size'] = 20000
    # Dataset generation
    audio_processor = dataset_copy.AudioProcessor(training_parameters, data_processing_parameters)
    # audio_processor.divide_data_set(training_parameters, task_id='dil_task_0')
    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    noise_types = ['TBUS_48k', 'OMEETING_48k', 'OHALLWAY_48k', 'NFIELD_48k', 'NPARK_48k', 'NRIVER_48k', \
                       'OOFFICE_48k', 'SPSQUARE_48k', 'DLIVING_48k', 'TCAR_48k', 'STRAFFIC_48k', 'PSTATION_48k', \
                       'DKITCHEN_48k', 'TMETRO_48k', 'DWASHING_48k', 'PCAFETER_48k', 'PRESTO_48k', 'SCAFE_48k' ]
        
    # TBUS_48k: 0, OMEETING_48k: 1, OHALLWAY_48k: 2, NFIELD_48k: 3, NPARK_48k: 4, NRIVER_48k: 5, ...
    noise_dict = {noise_types[i]: i for i in range(len(noise_types))}
    # print(noise_dict)
    

    data = dataset_copy.AudioGenerator('training', audio_processor, training_parameters, task_id=None, task = None)
    print("Data length: " + str(len(data)))
    num_iter = 20 if training_parameters['debug'] else len(data)
    inputs, labels, noises = data[0]
    inputs = inputs.to(device) # torch.Size([20000, 49, 10])
    labels = labels.to(device) # torch.Size([20000])
    # noises torch.Size([20000])

    # convert noise to integer according to noise_dict and convert to tensor
    noises = [noise_dict[noise] for noise in noises]
    noises = torch.Tensor(noises).to(device).long()
    import pandas as pd
    from scipy.stats import entropy
    # inputs.unsqueeze_(1)
    print('inputs size: ', inputs.size())

    # calculate probability distribution
    def compute_prob_distribution(mfcc):
        # convert to probability distribution
        prob_dist = torch.softmax(mfcc, dim=-1)
        return prob_dist

    # calculate entropy
    def compute_entropy(prob_dist):

        ent = entropy(prob_dist.cpu().numpy(), axis=-1)
        return ent

    # calculate average entropy per sample
    entropies = []
    for i in range(inputs.shape[0]):
        prob_dist = compute_prob_distribution(inputs[i])
        entropies.append(compute_entropy(prob_dist).mean())

    # convert list to tensor
    entropies = torch.tensor(entropies)

    # calculate average entropy per noise type
    def average_entropy_per_noise_type(entropies, noises):
        avg_entropies = []
        for noise_type in noises.unique():
            indices = (noises == noise_type)
            avg_entropies.append(entropies[indices].mean().item())
        return avg_entropies

    avg_entropies = average_entropy_per_noise_type(entropies, noises)

    # calculate KL divergence
    def compute_kl_divergence(prob_dist1, prob_dist2):
        kl_div = entropy(prob_dist1.cpu().numpy(), qk=prob_dist2.cpu().numpy(), axis=-1)
        return kl_div

    # calculate KL divergence matrix
    def compute_kl_divergence_matrix(inputs, noises):
        unique_noise_types = noises.unique()
        num_noise_types = unique_noise_types.size(0)
        kl_div_matrix = np.zeros((num_noise_types, num_noise_types))
        
        for i, noise_type1 in enumerate(unique_noise_types):
            for j, noise_type2 in enumerate(unique_noise_types):
                if i != j:
                    indices1 = (noises == noise_type1)
                    indices2 = (noises == noise_type2)
                    prob_dist1 = compute_prob_distribution(inputs[indices1]).mean(dim=0)
                    prob_dist2 = compute_prob_distribution(inputs[indices2]).mean(dim=0)
                    kl_div = compute_kl_divergence(prob_dist1, prob_dist2)
                    kl_div_matrix[i, j] = kl_div.mean()
        return kl_div_matrix

    kl_divergence_matrix = compute_kl_divergence_matrix(inputs, noises)

    # 输出平均熵和KL散度矩阵
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Average Entropies", dataframe=pd.DataFrame(avg_entropies))
    # tools.display_dataframe_to_user(name="KL Divergence Matrix", dataframe=pd.DataFrame(kl_divergence_matrix))
    print(kl_divergence_matrix)


    
    
  
  
  elif args.method == 'feat_vis_noise_data_test':

    print('Start feature visualization')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    n_classes = 18
    new_classes_per_task = 6
    task_id = 'dil_joint'
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=task_id)  # To be parametrized
    training_parameters['batch_size'] = 2048
    # Dataset generation
    audio_processor = dataset_copy.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)

    start=time.process_time()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    print('model loaded from ', model_path)
    training_environment.feat_vis_noise_test(model, task_id=task_id, statistics=False)


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
    if args.model_M:
      model = DSCNNM(use_bias = True, n_classes = n_classes)
    elif args.model_L:
      model = DSCNNL(use_bias = True, n_classes = n_classes)
    else:
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
    if args.model_M:
      model = DSCNNM(use_bias = True, n_classes = n_classes)
    elif args.model_L:
      model = DSCNNL(use_bias = True, n_classes = n_classes)
    else:
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
    print ("Testing Accuracy...")
    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    for i in range(4):
      #
      task_id = f'cil_task_{i}'
      acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
      print(f'Test Accuracy of Task {i}: ', acc_task)
      if args.wandb:
          wandb.log({f'ACC_task_{i}': acc_task})

      # del audio_processor
      # del training_environment
      

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
  

  elif args.method == 'NRS':

    print('Start NRS')

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
        total_time = 0
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            # memory_buffer.add_data(inputs_mb, labels_mb)
            start_time = time.time()
            memory_buffer.add_data(inputs_mb, labels_mb)
            end_time = time.time()
            total_time = total_time + (end_time - start_time)
            avg_time = total_time / (minibatch + 1)

            if args.wandb:
              cls_count = memory_buffer.get_class_count()
              # wandb.log({'Class count': cls_count})
              wandb.log({'AVG Time for adding data': avg_time})
            
            if minibatch % 50 == 0:
              # print average time
              print('Average time for adding data: ', avg_time)
                # cls_count = memory_buffer.get_class_count()
                # print("Class count: ", cls_count)
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

        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        
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


  elif args.method == 'ECBRS':

    print('Start ECBRS')

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
        memory_buffer = Buffer_ECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, 'cil_task_0', task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        total_time = 0
        for minibatch in range(int(len(data))):
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            if inputs_mb.size(0) == 0: # inputsize torch.Size([0, 1, 49, 10])
              break
            # memory_buffer.add_data(inputs_mb, labels_mb)
            start_time = time.time()
            memory_buffer.add_data(inputs_mb, labels_mb)
            end_time = time.time()
            total_time = total_time + (end_time - start_time)
            avg_time = total_time / (minibatch + 1)

            if args.wandb:
              cls_count = memory_buffer.get_class_count()
              # wandb.log({'Class count': cls_count})
              wandb.log({'AVG Time for adding data': avg_time})
            # print("Time for adding data: ", end_time - start_time)
            
            if minibatch % 10 == 0:
                cls_count = memory_buffer.get_class_count()
                # print("Class count: ", cls_count)
                print(f'adding {minibatch}/{len(data)} of data. Time for adding data: ')
                print("Average Time for adding data: ", avg_time)
            # print("Time for adding data: ", end_time - start_time)
            # if minibatch % 50 == 0:
            #     cls_count = memory_buffer.get_class_count()
            #     print("Class count: ", cls_count)
            #     print("Time for adding data: ", end_time - start_time)
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
        # delete data
        del data
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
        # count_ops(model, dummy_input)
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

        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
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


  elif args.method == 'DAECBRS':

    print('Start DAECBRS')

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
        memory_buffer = Buffer_DAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, 'cil_task_0', task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        total_time = 0
        for minibatch in range(int(len(data))):
            if args.debug and minibatch == 10:
              break
            if minibatch == int((17/35)*len(data)):
              break
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            # memory_buffer.add_data(inputs_mb, labels_mb)
            start_time = time.time()
            memory_buffer.add_data(inputs_mb, labels_mb)
            end_time = time.time()
            total_time = total_time + (end_time - start_time)
            avg_time = total_time / (minibatch + 1)

            if args.wandb:
              cls_count = memory_buffer.get_class_count()
              # wandb.log({'Class count': cls_count})
              wandb.log({'AVG Time for adding data': avg_time})
            # print("Time for adding data: ", end_time - start_time)
            
            if minibatch % 10 == 0:
                cls_count = memory_buffer.get_class_count()
                # print("Class count: ", cls_count)
                print(f'adding {minibatch}/{len(data)} of data. Time for adding data: ')
                print("Average Time for adding data: ", avg_time)
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
        # delete data
        del data
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
        # count_ops(model, dummy_input)
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

        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
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


  elif args.method == 'DAECBRS_entropy':

    print('Start DAECBRS_entropy')

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
        memory_buffer = Buffer_DAECB_entropy(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)
        # prepare data
        data = dataset.AudioGenerator('training', audio_processor, training_parameters, 'cil_task_0', task = None)
        # for minibatch in range(int(config['memory_buffer_size']/128)):
        total_time = 0
        for minibatch in range(int(len(data))):
        # for minibatch in range(1):
            if minibatch == int((17/35)*len(data)):
                break
            # return a random batch of data with batch size 128.
            inputs_mb, labels_mb, _ = data[0]
            # inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(device) # ([128, 1, 49, 10])
            # labels_mb = torch.Tensor(labels_mb).to(device).long() # ([128])
            inputs_mb.unsqueeze_(1)
            # memory_buffer.add_data(inputs_mb, labels_mb)
            start_time = time.time()
            memory_buffer.add_data(inputs_mb, labels_mb)
            end_time = time.time()
            total_time = total_time + (end_time - start_time)
            avg_time = total_time / (minibatch + 1)

            if args.wandb:
              cls_count = memory_buffer.get_class_count()
              # wandb.log({'Class count': cls_count})
              wandb.log({'AVG Time for adding data': avg_time})
            # print("Time for adding data: ", end_time - start_time)
            
            if minibatch % 10 == 0:
                cls_count = memory_buffer.get_class_count()
                # print("Class count: ", cls_count)
                print(f'adding {minibatch}/{len(data)} of data. Time for adding data: ')
                print("Average Time for adding data: ", avg_time)
        # print('Memory buffer initialized. Size:', memory_buffer.get_size())
        # delete data
        del data
        model = DSCNNS(use_bias = True, n_classes = n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        print('Model loaded from ', model_path)
        summary(model,(1,49,data_processing_parameters['feature_bin_count']))
        dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
        # count_ops(model, dummy_input)
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

        model, memory_buffer = training_environment.ER(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
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
  

  elif args.method == 'LAECBRS':

    print('Start LAECBRS')

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

      # task 0
      if i == 0:
        # train from scratch
        # initialize memory buffer
        memory_buffer = Buffer_LAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        print('Training model on cil_task_0...')

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

        # start=time.process_time()
        start_time_training = time.time()

        # Load the model and buffer checkpoint
        if args.load_buffer:

          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        
        # buffer will be initialized with task 0 data
        else:
          
          
          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          if args.save_buffer:
            # get everything from the buffer
            num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

            # make them a big dictionary
            buffer_state = {'num_seen_examples': num_seen_examples, 
                            'buffer': buffer, 
                            'loss': loss, 
                            'loss_index': loss_index, 
                            'class_count': class_count, 
                            'class_count_total': class_count_total, 
                            'full_classes': full_classes}
            # save buffer_state
            timestr = time.strftime("%Y%m%d-%H%M%S")
            buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
            buffer_path = './buffer_state/' + buffer_name
            torch.save(buffer_state, buffer_path)
            print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
          wandb.log({f'ACC_{task_id}': acc_task})
        
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

      # task 1, 2, 3
      else:
        memory_buffer.update_old_classes()

        # expand the fully connected layer for new classes
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

        # Prepare the training environment
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        
        # Start continual training
        model, memory_buffer = training_environment.ER_LAECB(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
        
        # Task-average Forgetting
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



  elif args.method == 'AAECBRS':

    print('Start AAECBRS')

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

      # task 0
      if i == 0:
        # train from scratch
        # initialize memory buffer
        memory_buffer = Buffer_AAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        print('Training model on cil_task_0...')

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

        # start=time.process_time()
        start_time_training = time.time()

        # Load the model and buffer checkpoint
        if args.load_buffer:

          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        
        # buffer will be initialized with task 0 data
        else:
          
          
          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_AAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_AAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          if args.save_buffer:
            # get everything from the buffer
            num_seen_examples, buffer, acc, acc_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

            # make them a big dictionary
            buffer_state = {'num_seen_examples': num_seen_examples, 
                            'buffer': buffer, 
                            'acc': acc, 
                            'ac_index': acc_index, 
                            'class_count': class_count, 
                            'class_count_total': class_count_total, 
                            'full_classes': full_classes}
            # save buffer_state
            timestr = time.strftime("%Y%m%d-%H%M%S")
            buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
            buffer_path = './buffer_state/' + buffer_name
            torch.save(buffer_state, buffer_path)
            print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
          wandb.log({f'ACC_{task_id}': acc_task})
        
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

      # task 1, 2, 3
      else:
        # memory_buffer.update_old_classes()

        # expand the fully connected layer for new classes
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

        # Prepare the training environment
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        
        # Start continual training
        model, memory_buffer = training_environment.ER_AAECB(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
        
        # Task-average Forgetting
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


  elif args.method == 'GAECBRS':

    print('Start GAECBRS')

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

      # task 0
      if i == 0:
        # train from scratch
        # initialize memory buffer
        memory_buffer = Buffer_GAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        print('Training model on cil_task_0...')

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

        # start=time.process_time()
        start_time_training = time.time()

        # Load the model and buffer checkpoint
        if args.load_buffer:

          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        
        # buffer will be initialized with task 0 data
        else:
          # empty cache
          torch.cuda.empty_cache()
          
          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_GAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_GAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          if args.save_buffer:
            # get everything from the buffer
            num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

            # make them a big dictionary
            buffer_state = {'num_seen_examples': num_seen_examples, 
                            'buffer': buffer, 
                            'loss': loss, 
                            'loss_index': loss_index, 
                            'class_count': class_count, 
                            'class_count_total': class_count_total, 
                            'full_classes': full_classes}
            # save buffer_state
            timestr = time.strftime("%Y%m%d-%H%M%S")
            buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
            buffer_path = './buffer_state/' + buffer_name
            torch.save(buffer_state, buffer_path)
            print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
          wandb.log({f'ACC_{task_id}': acc_task})
        
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

      # task 1, 2, 3
      else:
        memory_buffer.update_old_classes()

        # expand the fully connected layer for new classes
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

        # Prepare the training environment
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        
        # Start continual training
        model, memory_buffer = training_environment.ER_GAECB(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
        
        # Task-average Forgetting
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


  elif args.method == 'LDAECBRS':

    print('Start LDAECBRS')

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

      # task 0
      if i == 0:
        # train from scratch
        # initialize memory buffer
        memory_buffer = Buffer_LDAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        print('Training model on cil_task_0...')

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

        # start=time.process_time()
        start_time_training = time.time()

        # Load the model and buffer checkpoint
        if args.load_buffer:

          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        
        # buffer will be initialized with task 0 data
        else:
          
          
          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_LDAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_LDAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          # get everything from the buffer
          num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

          # make them a big dictionary
          buffer_state = {'num_seen_examples': num_seen_examples, 
                          'buffer': buffer, 
                          'loss': loss, 
                          'loss_index': loss_index, 
                          'class_count': class_count, 
                          'class_count_total': class_count_total, 
                          'full_classes': full_classes}
          # save buffer_state
          timestr = time.strftime("%Y%m%d-%H%M%S")
          buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
          buffer_path = './buffer_state/' + buffer_name
          torch.save(buffer_state, buffer_path)
          print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
          wandb.log({f'ACC_{task_id}': acc_task})
        
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

      # task 1, 2, 3
      else:
        memory_buffer.update_old_classes()

        # expand the fully connected layer for new classes
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

        # Prepare the training environment
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        
        # Start continual training
        model, memory_buffer = training_environment.ER_LDAECB(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
        
        # Task-average Forgetting
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


  elif args.method == 'LDAAECBRS':

    print('Start LDAAECBRS')

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

      # task 0
      if i == 0:
        # train from scratch
        # initialize memory buffer
        memory_buffer = Buffer_LDAAECB(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        print('Training model on cil_task_0...')

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

        # start=time.process_time()
        start_time_training = time.time()

        # Load the model and buffer checkpoint
        if args.load_buffer:

          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        
        # buffer will be initialized with task 0 data
        else:
          
          
          if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print('Model loaded from ', model_path)
            _, memory_buffer = training_environment.ER_LDAAECB(model, memory_buffer, task_id)
            # restore the model to the state before the 1-epoch training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          else: # Buffer is initialized with task 0 data from-scratch trained model.
            model_path, memory_buffer = training_environment.ER_LDAAECB(model, memory_buffer, task_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          # get everything from the buffer
          num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

          # make them a big dictionary
          buffer_state = {'num_seen_examples': num_seen_examples, 
                          'buffer': buffer, 
                          'loss': loss, 
                          'loss_index': loss_index, 
                          'class_count': class_count, 
                          'class_count_total': class_count_total, 
                          'full_classes': full_classes}
          # save buffer_state
          timestr = time.strftime("%Y%m%d-%H%M%S")
          buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
          buffer_path = './buffer_state/' + buffer_name
          torch.save(buffer_state, buffer_path)
          print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
          wandb.log({f'ACC_{task_id}': acc_task})
        
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

      # task 1, 2, 3
      else:
        memory_buffer.update_old_classes()

        # expand the fully connected layer for new classes
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

        # Prepare the training environment
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        
        # Start continual training
        model, memory_buffer = training_environment.ER_LDAAECB(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
        
        # Task-average Forgetting
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



  elif args.method == 'DAECBRS_reg':

    print('Start DAECBRS_reg')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    # tasks = ['cil_task_1', 'cil_task_2', 'cil_task_3']
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

      # task 0
      if i == 0:
        # train from scratch
        # initialize memory buffer
        memory_buffer = Buffer_DAECB_reg(buffer_size=config['memory_buffer_size'], batch_size=config['batch_size'], device=device)

        print('Training model on cil_task_0...')

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

        # start=time.process_time()
        start_time_training = time.time()

        # Load the model and buffer checkpoint
        if args.load_buffer:

          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          
          buffer_path = './buffer_state/' + buffer_name
          buffer_state = torch.load(buffer_path)
          # print(buffer_state)
          print('Loaded buffer state from:', buffer_path)
          # print('buffer state:' , buffer_state)
          if buffer_path is not None:
            buffer_ckpt = torch.load(buffer_path)
            memory_buffer.load_buffer(buffer_ckpt)
          else: 
            print('No buffer saved')
            break
        
        # buffer will be initialized with task 0 data
        else:
          
          
          # if args.load_model: # Buffer is initialized with a 1-epoch training on task 0 using model checkpoint
          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          print('Model loaded from ', model_path)
          _, memory_buffer = training_environment.ER_DAECB_reg(model, memory_buffer, task_id)
          # restore the model to the state before the 1-epoch training
          model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
          # else: # Buffer is initialized with task 0 data from-scratch trained model.
          #   model_path, memory_buffer = training_environment.ER_DAECB_reg(model, memory_buffer, task_id)
          #   model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

          # # get everything from the buffer
          # num_seen_examples, buffer, loss, loss_index, class_count, class_count_total, full_classes = memory_buffer.get_entire_buffer()

          # # make them a big dictionary
          # buffer_state = {'num_seen_examples': num_seen_examples, 
          #                 'buffer': buffer, 
          #                 'loss': loss, 
          #                 'loss_index': loss_index, 
          #                 'class_count': class_count, 
          #                 'class_count_total': class_count_total, 
          #                 'full_classes': full_classes}
          # # save buffer_state
          # timestr = time.strftime("%Y%m%d-%H%M%S")
          # buffer_name = 'buffer_'+ args.mode + '_' + args.method + '_' + timestr + '.pth'
          # buffer_path = './buffer_state/' + buffer_name
          # torch.save(buffer_state, buffer_path)
          # print(f'Buffer state saved at {buffer_path} as {buffer_name}')

          # print('Finished Training on GPU in {:.2f} seconds'.format(time.time()-start_time_training))
        
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id='cil_task_0', statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
          wandb.log({f'ACC_{task_id}': acc_task})
        
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

      # task 1, 2, 3
      else:
        memory_buffer.update_old_classes()

        # expand the fully connected layer for new classes
        fc_new= torch.nn.Linear(64, n_classes + new_classes_per_task) # 19 + 6 = 25, 19 + 12 = 31, 19 + 18 = 37
        if model.fc1 is not None:
          weight = copy.deepcopy(model.fc1.weight.data)
          bias = copy.deepcopy(model.fc1.bias.data)
          # copy old weights and biases to fc_new
          fc_new.weight.data[:n_classes] = weight
          fc_new.bias.data[:n_classes] = bias
          # replace fc1 with fc_new
          model.fc1 = fc_new

        # Prepare the training environment
        training_environment = Train(audio_processor, training_parameters, model, device, args, config)
        print(f'Conintual Training on {task_id}...')
        
        # Start continual training
        model, memory_buffer = training_environment.ER_DAECB_reg(model, memory_buffer, task_id)
        cls_count = memory_buffer.get_class_count()
        print("Class count: ", cls_count)
        acc_task = training_environment.validate(model, mode='testing', batch_size=-1, task_id=task_id, statistics=False)
        print(f'Test Accuracy of {task_id}: ', acc_task)
        if args.wandb:
            wandb.log({f'ACC_{task_id}': acc_task})
        n_classes += new_classes_per_task
        
        # Task-average Forgetting
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




  elif args.method == 'feat_vis':

    print('Start feature visualization')

    acc_matrix = np.zeros((4,4)) # acc_matrix[i,j] = acc after training task i on task j
    acc_done = 0

    tasks = ['cil_task_0','cil_task_1', 'cil_task_2', 'cil_task_3']
    n_classes = 37
    new_classes_per_task = 6
    task_id = 'cil_task_3'
    training_parameters, data_processing_parameters = parameter_generation(args, config, task_id=None)  # To be parametrized
    training_parameters['batch_size'] = 2048
    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

    model = DSCNNS(use_bias = True, n_classes = n_classes) # 35 words
    model.to(device)
    summary(model,(1,49,data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, args, config)

    start=time.process_time()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    print('model loaded from ', model_path)
    training_environment.feat_vis(model, task_id=task_id, statistics=False)

 
