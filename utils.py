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


def task_average_forgetting(acc_matrix):

    acc_final = acc_matrix[-1,:]
    acc_best = np.max(acc_matrix, axis = 0)
    forgetting = acc_best - acc_final
    average_forgetting = np.mean(forgetting)/100

    return average_forgetting


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



def parameter_generation(args=None, config=None, task_id=None):
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
        # Task 2: keyword 24 to 29
        # Task 3: keyword 30 to 35
        if args.mode == 'dil':

            if task_id == None:
                # not used.
                training_parameters['noise_test']  = ['DKITCHEN']

                training_parameters['noise_train'] = ['DKITCHEN']

            elif task_id == 'dil_task_0':
                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
                
                training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']

            elif task_id == 'dil_task_1':

                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'PCAFETER', 'PRESTO', 'PSTATION']
                
                # training_parameters['noise_test'] = ['DKITCHEN']

                training_parameters['noise_train'] = ['PCAFETER', 'PRESTO', 'PSTATION']

            elif task_id == 'dil_task_2':
                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'PCAFETER', 'PRESTO', 'PSTATION',\
                                                    'SCAFE', 'SPSQUARE', 'STRAFFIC']

                training_parameters['noise_train'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC']
                
            elif task_id == 'dil_task_3':
                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'PCAFETER', 'PRESTO', 'PSTATION',\
                                                    'SCAFE', 'SPSQUARE', 'STRAFFIC',\
                                                    'TBUS', 'TCAR', 'TMETRO']

                training_parameters['noise_train'] = ['TBUS', 'TCAR', 'TMETRO']

            elif task_id == 'dil_joint':
                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'PCAFETER', 'PRESTO', 'PSTATION',\
                                                    'SCAFE', 'SPSQUARE', 'STRAFFIC',\
                                                    'TBUS', 'TCAR', 'TMETRO']

                training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'PCAFETER', 'PRESTO', 'PSTATION',\
                                                    'SCAFE', 'SPSQUARE', 'STRAFFIC',\
                                                    'TBUS', 'TCAR', 'TMETRO']
                
            if task_id == 'dil_task_0_disjoint':
                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.
            
            elif task_id == 'dil_task_1_disjoint':
                training_parameters['noise_test']  = ['PCAFETER', 'PRESTO', 'PSTATION']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.

            elif task_id == 'dil_task_2_disjoint':
                training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.

            elif task_id == 'dil_task_3_disjoint':
                training_parameters['noise_test']  = ['TBUS', 'TCAR', 'TMETRO']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.
            
        else:
            # not used.
            training_parameters['noise_test'] = ['DKITCHEN']

            training_parameters['noise_train'] = ['DKITCHEN']       
            
    else:
        training_parameters['noise_dir'] = config['data_dir']+'/_background_noise_'

    # target_words='yes,no,up,down,left,right,on,off,stop,go,'  # GSCv2 - 10 words
                 #2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11
    target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words


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


class Buffer_NRS:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.num_seen_examples = 0
        self.buffer = {}
        self.attributes = ['examples', 'labels']
        
        self.buffer['examples'] = torch.empty((self.buffer_size, 1, 49, 10), device=self.device)
        self.buffer['labels'] = torch.empty((self.buffer_size), device=self.device)
        print("Buffer initialized")
    
    def to_device(self, model_device):
        # Move the buffer to the device of the model
        self.buffer['examples'] = self.buffer['examples'].to(model_device)
        self.buffer['labels'] = self.buffer['labels'].to(model_device)

    def naive_reservoir(self) -> int:
        """
        Naive Reservoir Sampling algorithm.

        """
        # if self.num_seen_examples < self.buffer_size:
        #     return self.num_seen_examples

        rand = np.random.randint(0, self.num_seen_examples + 1)
        if rand < self.buffer_size:
            return rand
        else:
            return -1

    def add_data(self, examples, labels):
        """
        Add data to the buffer.
        examples: torch.Size([128, 1, 49, 10])
        labels: torch.Size([128])
        """
        input_size = examples.size(0)
        current_buffer_size = self.buffer['examples'].size(0)
        # if buffer is not full, add batch data to buffer
        if current_buffer_size < self.buffer_size:
            
            self.buffer['examples'][current_buffer_size:current_buffer_size + input_size] = examples
            self.buffer['labels'][current_buffer_size:current_buffer_size + input_size] = labels
            self.num_seen_examples += input_size
            # print("Data added to buffer")
        else:
            # assert current_buffer_size == self.buffer_size
            for i in range(input_size):
                sample_index = self.naive_reservoir()
                # print(sample_index)
                if sample_index != -1:
                    self.buffer['examples'][sample_index] = examples[i]
                    self.buffer['labels'][sample_index] = labels[i]
                    # print("Data added to buffer")
                    self.num_seen_examples += 1

    def get_data(self):
        """
        Get data from the buffer.
        """
        # indices = torch.randperm(self.num_seen_examples)[:self.batch_size]
        # return self.buffer['examples'][indices], self.buffer['labels'][indices]
        indices = torch.randperm(self.buffer_size).to(self.buffer['examples'].device)[:self.batch_size]
        # print(indices)
        return self.buffer['examples'][indices], self.buffer['labels'][indices]
        
    
    def get_size(self):
        """
        Get the number of examples in the buffer.
        """
        num_examples = self.buffer['examples'].size(0)
        num_labels = self.buffer['labels'].size(0)
        assert num_examples == num_labels
        return num_examples

    def reset_num_seen_examples(self):
        """
        Reset the number of seen examples.
        """
        self.num_seen_examples = 0

    def is_empty(self):
        """
        Check if the buffer is empty.
        """
        return self.num_seen_examples == 0  
    
    def get_class_count(self):
        """
        Get the number of examples for each class in the buffer.
        """
        class_count = {}
        for label in self.buffer['labels']:
            if label.item() in class_count:
                class_count[label.item()] += 1
            else:
                class_count[label.item()] = 1
        return class_count

class Buffer_CB:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.num_seen_examples = 0
        self.buffer = {}
        self.attributes = ['examples', 'labels']
        self.num_classes_init = 19
        self.num_new_classes = 6
        self.buffer['examples'] = torch.empty((self.buffer_size, 1, 49, 10), device=self.device)
        self.buffer['labels'] = torch.empty((self.buffer_size), device=self.device)

        # devide the buffer into 19 parts, each part for one class. Dont actually devide the buffer,
        # but mark the start and end of each class in the buffer.
    
        print("Buffer initialized")
    
    def naive_reservoir(self) -> int:
        """
        Naive Reservoir Sampling algorithm.

        """
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples

        rand = np.random.randint(0, self.num_seen_examples + 1)
        if rand < self.buffer_size:
            return rand
        else:
            return -1

    def add_data(self, examples, labels):
        """
        Add data to the buffer.
        examples: torch.Size([128, 1, 49, 10])
        labels: torch.Size([128])
        """
        input_size = examples.size(0)
        current_buffer_size = self.buffer['examples'].size(0)
        # if buffer is not full, add batch data to buffer
        if current_buffer_size < self.buffer_size:
            
            self.buffer['examples'][current_buffer_size:current_buffer_size + input_size] = examples
            self.buffer['labels'][current_buffer_size:current_buffer_size + input_size] = labels
            self.num_seen_examples += input_size
            # print("Data added to buffer")
        else:
            assert current_buffer_size == self.buffer_size
            for i in range(input_size):
                sample_index = self.naive_reservoir()
                if sample_index != -1:
                    self.buffer['examples'][sample_index] = examples[i]
                    self.buffer['labels'][sample_index] = labels[i]
                    # print("Data added to buffer")
                    self.num_seen_examples += 1

    def get_data(self):
        """
        Get data from the buffer.
        """
        indices = torch.randperm(self.num_seen_examples)[:self.batch_size]
        return self.buffer['examples'][indices], self.buffer['labels'][indices]
        
    
    def get_size(self):
        """
        Get the number of examples in the buffer.
        """
        num_examples = self.buffer['examples'].size(0)
        num_labels = self.buffer['labels'].size(0)
        assert num_examples == num_labels
        return num_examples

    def reset_num_seen_examples(self):
        """
        Reset the number of seen examples.
        """
        self.num_seen_examples = 0

    def is_empty(self):
        """
        Check if the buffer is empty.
        """
        return self.num_seen_examples == 0  
    
    def get_class_count(self):
        """
        Get the number of examples for each class in the buffer.
        """
        class_count = {}
        for label in self.buffer['labels']:
            if label.item() in class_count:
                class_count[label.item()] += 1
            else:
                class_count[label.item()] = 1
        return class_count