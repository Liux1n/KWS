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
print (torch.version.__version__)
print(device)



parser = argparse.ArgumentParser()

parser.add_argument('--training_mode', choices=['base', 'joint', 'dil', 'cil'], default='add',
                    help='training mode (default: base)')
parser.add_argument('--noise_mode', choices=['nlkws', 'nakws', 'odda'], default='add',
                    help='noise mode')
parser.add_argument('--debug', action='store_true', help="Enable debug mode.")

args = parser.parse_args()

config = load_config("config.yaml")

wandb.init(project=config['project_name'], entity="liux1n", config=config)

if args.training_mode == 'base':
   print('Base model')

# Parameter generation
training_parameters, data_processing_parameters = parameter_generation(args, config)  # To be parametrized

# Dataset generation
audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

train_size = audio_processor.get_size('training')
valid_size = audio_processor.get_size('validation')
test_size = audio_processor.get_size('testing')
print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

# Model generation and analysis
model = DSCNNS(use_bias = True, n_classes = 37)
model.to(device)
summary(model,(1,49,data_processing_parameters['feature_bin_count']))
dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
# count_ops(model, dummy_input)

# Training initialization
training_environment = Train(audio_processor, training_parameters, model, device)

# Removing stored inputs and activations
remove_txt()

if args.training_mode == 'base':
  train = True
if train:
  # Train
  # start=time.clock_gettime(0)
  start=time.process_time()
  
  training_environment.train(model)
  print('Finished Training on GPU in {:.2f} seconds'.format(time.process_time()-start))

# Perform ODDA
odda = False
if odda:
  # Load pretrained model
  model.load_state_dict(torch.load('./pretrain2_ordered_v3_fixFilfixUttr.pth', map_location=torch.device('cuda')))
  environmental_noise = 'TMETRO'
  training_environment.adapt(model, environmental_noise)

# # Accuracy on the training set. 
# print ("Training acc")
# acc = training_environment.validate(model, mode='training', batch_size=-1, statistics=False)
# Accuracy on the validation set. 
# print ("Validation acc")
# acc = training_environment.validate(model, mode='validation', batch_size=-1, statistics=False)
# Accuracy on the testing set. 
print ("Testing acc")
acc = training_environment.validate(model, mode='testing', batch_size=-1, statistics=False)

