from methods.base_pretrain import base_pretrain
from methods.joint_pretrain import joint_pretrain
from methods.source_only import source_only
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

from utils import load_config
from torchsummary import summary
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
                  choices=['source_only', 'finetune', 'joint', 'ER_NRS', 'ECBRS', 'ER_CB', 'ER_CB_fast', 'ER_ECB', 'ER_LAECB',
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
model_name = 'dil_base_pretrain_vol_5.pth'
# model_name = 'dil_base_pretrain_vol_10.pth'
# model_name = 'dil_joint_pretrain_vol_10.pth'

# CIL joint
# model_name = 'cil_joint_pretrain_20240418-23171014.pth'

# CIL base pretrain
# model_name = 'cil_base_pretrain_20240419-02554315.pth'


# model_name = 'joint_12_cil_joint_model.pth'

model_path = './models/' + model_name 

methods = {
    'base_pretrain': base_pretrain,
    'joint_pretrain': joint_pretrain,
    'source_only': source_only
}

Trainer = methods[args.method](model_path, device, args, config)

Trainer.run()

