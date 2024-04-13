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


import os

from sklearn.metrics import confusion_matrix

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from typing import Tuple
from torchvision import transforms

def npy_to_txt(layer_number, activations):
    # Saving the input

    if layer_number == -1:
        tmp = activations.reshape(-1)
        f = open('input.txt', "a")
        f.write('# input (shape [1, 49, 10]),\\\n')
        for elem in tmp:
            if (elem < 0):
                f.write (str(256+elem) + ",\\\n")
            else:
                f.write (str(elem) + ",\\\n")
        f.close()
    # Saving layers' activations
    else:
        tmp = activations.reshape(-1)
        f = open('out_layer' + str(layer_number) + '.txt', "a")
        f.write('layers.0.relu1 (shape [1, 25, 5, 64]),\\\n')  # Hardcoded, should be adapted for better understanding.
        for elem in tmp:
            if (elem < 0):
                f.write (str(256+elem) + ",\\\n")
            else:
                f.write (str(elem) + ",\\\n")
        f.close()


def remove_txt():
    # Removing old activations and inputs

    directory = '.'
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if (file.startswith("out_layer") or file.startswith("input.txt"))]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


def conf_matrix(labels, predicted, training_parameters):
    # Plotting confusion matrix

    labels = labels.cpu()
    predicted = predicted.cpu()
    cm = confusion_matrix(labels, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = [i for i in ['silence','unknown']+training_parameters['wanted_words']],
                  columns = [i for i in ['silence','unknown']+training_parameters['wanted_words']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def per_noise_accuracy(labels, predicted, noises):

    noise_types = list(set(noises))

    for noise in noise_types:
        correct = 0
        total = 0

        for i in range (0, len(noises)):
            if (noises[i] == noise):
                total = total + 1
                if ((labels == predicted)[i]):
                    correct = correct + 1
        print('Noise number %3d - accuracy: %.3f' % (noise,  100 * correct / total))   

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)



def parameter_generation(args=None,config=None):
    # Data processing parameters

    config = load_config("config.yaml")

    data_processing_parameters = {
    'feature_bin_count':10
    }
    time_shift_ms=200
    sample_rate=16000
    clip_duration_ms=1000
    time_shift_samples= int((time_shift_ms * sample_rate) / 1000)
    window_size_ms=40.0
    window_stride_ms=20.0
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    data_processing_parameters['desired_samples'] = desired_samples
    data_processing_parameters['sample_rate'] = sample_rate
    data_processing_parameters['spectrogram_length'] = spectrogram_length
    data_processing_parameters['window_stride_samples'] = window_stride_samples
    data_processing_parameters['window_size_samples'] = window_size_samples

    # Training parameters
    training_parameters = {
    'noise_mode':args.noise_mode, # nlkws, nakws, odda
    'noise_dataset':config['noise_dataset'],
    'data_dir':config['data_dir'],
    # 'data_url':config['data_url'],
    'epochs':config['epochs'],
    'batch_size':config['batch_size'],
    'silence_percentage':config['silence_percentage'],
    'unknown_percentage':config['unknown_percentage'],
    'validation_percentage':config['validation_percentage'],
    'testing_percentage':config['testing_percentage'],
    'background_frequency':config['background_frequency'],
    'background_volume':config['background_volume'],
    'debug':args.debug,
    'wandb':args.wandb,
    }

    if training_parameters['noise_dataset'] == 'demand':
        training_parameters['noise_dir'] = config['noise_dir']
        # task:
        # choices=['dil_task_0', 'dil_task_1' , 'dil_task_2', 'dil_task_3' , 'dil_joint',
        #          'cil_task_0', 'cil_task_1', 'cil_task_2', 'cil_task_3', 'cil_joint'
        #          ], 
        # DIL:
        # Task 0: noise 1 to 9
        # Task 1: noise aug 10 to 12
        # Task 2: noise aug 13 to 15
        # Task 3: noise aug 16 to 18
        # CIL: 
        # Task 0: keyword 1 to 17
        # Task 1: keyword 18 to 23
        # Task 2: keyword 19 to 29
        # Task 3: keyword 30 to 35

        if args.task == 'dil_task_0':
            training_parameters['noise_test'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
            
            training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']

        elif args.task == 'dil_task_1':

            training_parameters['noise_test'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                'PCAFETER', 'PRESTO', 'PSTATION']
            
            # training_parameters['noise_test'] = ['DKITCHEN']

            training_parameters['noise_train'] = ['PCAFETER', 'PRESTO', 'PSTATION']

        elif args.task == 'dil_task_2':
            training_parameters['noise_test'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                'PCAFETER', 'PRESTO', 'PSTATION',\
                                                'SCAFE', 'SPSQUARE', 'STRAFFIC']

            training_parameters['noise_train'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC']
            
        elif args.task == 'dil_task_3':
            training_parameters['noise_test'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                'PCAFETER', 'PRESTO', 'PSTATION',\
                                                'SCAFE', 'SPSQUARE', 'STRAFFIC',\
                                                'TBUS', 'TCAR', 'TMETRO']

            training_parameters['noise_train'] = ['TBUS', 'TCAR', 'TMETRO']

        if args.task == 'dil_joint':
            training_parameters['noise_test'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                'PCAFETER', 'PRESTO', 'PSTATION',\
                                                'SCAFE', 'SPSQUARE', 'STRAFFIC',\
                                                'TBUS', 'TCAR', 'TMETRO']

            training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                'PCAFETER', 'PRESTO', 'PSTATION',\
                                                'SCAFE', 'SPSQUARE', 'STRAFFIC',\
                                                'TBUS', 'TCAR', 'TMETRO']
            
            

    else:
        training_parameters['noise_dir'] = config['data_dir']+'/_background_noise_'

    # target_words='yes,no,up,down,left,right,on,off,stop,go,'  # GSCv2 - 10 words
                 #2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11
    # Selecting 35 words
    
    if args.task == 'dil_task_0' or args.task == 'dil_task_1' or \
       args.task == 'dil_task_2' or args.task == 'dil_task_3' or args.task == 'dil_joint':
        target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
    elif args.task == 'cil_task_0':
        target_words='yes,no,up,down,left,\
                      right,on,off,stop,go,\
                      backward,bed,bird,cat,dog,\
                      eight,five,'  # GSCv2 - 17 words

    wanted_words=(target_words).split(',')
    wanted_words.pop()
    training_parameters['wanted_words'] = wanted_words
    training_parameters['time_shift_samples'] = time_shift_samples

    return training_parameters, data_processing_parameters


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, fsr=False, current_task=0) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """

        if size > self.examples.shape[0]:
            size = self.examples.shape[0]
        if fsr and current_task > 0:
            past_examples = self.examples[self.task_labels != current_task]
            if size > past_examples.shape[0]:
                size = past_examples.shape[0]
            if past_examples.shape[0]:
                choice = np.random.choice(min(self.num_seen_examples, past_examples.shape[0]), size=size,
                                          replace=False)
                if transform is None: transform = lambda x: x
                ret_tuple = (torch.stack([transform(ee.cpu())
                                          for ee in past_examples[choice]]).to(
                    self.device),)
                for attr_str in self.attributes[1:]:
                    if hasattr(self, attr_str):
                        attr = getattr(self, attr_str)
                        ret_tuple += (attr[self.task_labels != current_task][choice],)
            else: return tuple([torch.tensor([0])] * 4)
        else:
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), size=min(self.num_seen_examples, size),
                                      replace=False)
            if transform is None: transform = lambda x: x
            ret_tuple = (torch.stack([transform(ee.cpu())
                                for ee in self.examples[choice]]).to(self.device),)
            for attr_str in self.attributes[1:]:
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

