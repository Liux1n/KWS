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
from scipy.io.wavfile import write
import copy
import torch
import dataset
import dataset_test
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
                  Buffer_NRS, Buffer_ECB, Buffer_DAECB, Buffer_LAECB, Buffer_ClusterECB, Buffer_LossECB, Buffer_LDAECB
from copy import deepcopy
from pthflops import count_ops
from train import Train
import argparse

# from methods import base_pretrain, joint_pretrain


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.cuda.set_device(2)

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
                           'base_pretrain', 'joint_pretrain', 'custom', 'LossECB', 'noise_data_test', 'LDAECBRS'
                           ], 
                  default='base_pretrain',
                  required=True,
                  help='training  method (default: base)')

parser.add_argument('--noise_mode', choices=['nlkws', 'nakws', 'odda'], default='nakws',
                    required=True,
                    help='noise mode')

parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
parser.add_argument('--wandb', action='store_true', help="Enable wandb log.")
parser.add_argument('--background_volume', type=int, help="Set background volume.")
parser.add_argument('--early_stop', action='store_true', help="Enable early_stop.")
parser.add_argument('--forgetting', action='store_true', help="Enable forgetting measure.")
parser.add_argument('--dynamic_loss', action='store_true', help="Enable dynamic_loss.")
parser.add_argument('--augmentation', action='store_true', help="Enable data augmentation.")
parser.add_argument('--remark', type=str, help='Remark of the run.')
parser.add_argument('--epoch', type=int, help='Number of epochs for training.')
parser.add_argument('--epoch_cl', type=int, help='Number of epochs for continual training.')
parser.add_argument('--n_clusters', type=int, help='Number of clusters in ClusterECBRS.')
parser.add_argument('--load_buffer', action='store_true', help="Enable load_buffer.")
parser.add_argument('--load_model', action='store_true', help="Enable load_model.")
parser.add_argument('--seed', type=int, help='Number of epochs for training.')
parser.add_argument('--snr', type=int, help='Number of epochs for training.')
args = parser.parse_args()


time_str = time.strftime("%Y%m%d-%H%M%S")
if args.epoch:
  config['epochs'] = args.epoch
  print('Number of epochs changed to ', args.epoch)

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


training_parameters, data_processing_parameters = parameter_generation(args, config, task_id='dil_test')
# 'DKITCHEN', 'DLIVING', 'DWASHING', 
# 'NFIELD', 'NPARK', 'NRIVER', 
# 'OHALLWAY', 'OMEETING', 'OOFFICE',
# 'PCAFETER', 'PRESTO', 'PSTATION',
# 'SCAFE', 'SPSQUARE', 'STRAFFIC',
# 'TBUS', 'TCAR', 'TMETRO'
noise = 'STRAFFIC'
training_parameters['noise_train'] = [f'{noise}']
training_parameters['noise_test'] = [f'{noise}']
audio_processor = dataset_test.AudioProcessor(training_parameters, data_processing_parameters)
data = dataset_test.AudioGenerator('training', audio_processor, training_parameters, 'dil_test', task = None)

input, _, _ = data[0] 
sample_input = input[2] # input.shape (16000,)
print('input.shape', sample_input.shape)
# convert the input back to .wav file
snr = config['snr']
sample_rate = 16000
out_path = '/usr/scratch/sassauna2/sem24f39/KWS/audio/test_audio_' + noise + '_' + str(snr) + 'dB.wav'
write(out_path, sample_rate, sample_input)
print('Audio file is saved at', out_path)
# write('output.wav', sample_rate, sample_input)
