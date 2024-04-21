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
import random
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

    def reservoir(self) -> int:
        """
        Reservoir Sampling algorithm.

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
                sample_index = self.reservoir()
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
        self.buffer_filled = False
        self.buffer['examples'] = torch.empty((self.buffer_size, 1, 49, 10), device=self.device)
        self.buffer['labels'] = torch.empty((self.buffer_size), device=self.device)

        self.example_count_per_class = {}

        self.full_classes = []
        print("Buffer initialized")
    
    def to_device(self, model_device):
        # Move the buffer to the device of the model
        self.buffer['examples'] = self.buffer['examples'].to(model_device)
        self.buffer['labels'] = self.buffer['labels'].to(model_device)

    def reservoir(self) -> int:
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
        
        if self.num_seen_examples < self.buffer_size:
    
            self.buffer['examples'][self.num_seen_examples:self.num_seen_examples + input_size] = examples
            self.buffer['labels'][self.num_seen_examples:self.num_seen_examples + input_size] = labels
            self.num_seen_examples += input_size
            # print("Data added to buffer")
            class_count = self.get_class_count()
            
            for cls, count in class_count.items():
                if cls in self.example_count_per_class:
                    self.example_count_per_class[cls] += count
                else:
                    self.example_count_per_class[cls] = count
            
        else:
            
            for i in range(input_size):
                class_count = self.get_class_count()

                largest_class_count = max(class_count.values())

                largest_classes = set(int(cls) for cls, count in class_count.items() if count == largest_class_count)
                largest_classes_tensor = torch.tensor(list(largest_classes)).to(self.device)
                # temp_full_classes = self.full_classes
                self.full_classes = list(set(self.full_classes).union(largest_classes))
                # if label not in largest_classes: 
                label = labels[i].item()

                if label not in self.full_classes:

                    mask = (self.buffer['labels'].unsqueeze(1) == largest_classes_tensor).any(1)
                    indices_of_largest_classes = torch.where(mask)[0]
                    random_examples_idx = int(torch.randperm(indices_of_largest_classes.size(0))[0])

                    # take the first element of random_examples_idx
                    self.buffer['examples'][random_examples_idx] = examples[i]
                    self.buffer['labels'][random_examples_idx] = labels[i].float()
 
                    # add class count
                    if label in self.example_count_per_class:
                        self.example_count_per_class[label] += 1
                    else:
                        self.example_count_per_class[label] = 1

                    self.num_seen_examples += 1
                    
                else:
                    count_label_current = class_count[label]
                    count_label_total = self.example_count_per_class[label]
                    
                    # sample a u from uniform[0,1]
                    u = int(torch.rand(1))
                    if u <= count_label_current / count_label_total:
                        # take idx of examples in self.buffer['labels'] with label is in largest_classes
                        largest_classes_tensor = torch.tensor(list(largest_classes)).to(self.device)
                        mask = (self.buffer['labels'].unsqueeze(1) == largest_classes_tensor).any(1)
                        indices_of_largest_classes = torch.where(mask)[0]
                        random_examples_idx = int(torch.randperm(indices_of_largest_classes.size(0))[0])
                        self.buffer['examples'][random_examples_idx] = examples[i]
                        self.buffer['labels'][random_examples_idx] = labels[i].float()
                        # add class count
                        if label in self.example_count_per_class:
                            self.example_count_per_class[label] += 1
                        else:
                            self.example_count_per_class[label] = 1
                        self.num_seen_examples += 1
                        
                    else:
                        pass
            
    
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

class Buffer_CB_fast:
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
        self.buffer_filled = False
        # self.buffer['examples'] = torch.empty((self.buffer_size, 1, 49, 10), device=self.device)
        # self.buffer['labels'] = torch.empty((self.buffer_size), device=self.device)

        self.class_count = {}

        self.class_count_total = {}

        self.full_classes = []
        print("Buffer initialized")


    def get_total_class_count(self):
        """
        Get the total number of examples for each class in the self.buffer[label]
        """
        total_class_count = 0
        for label in self.buffer.keys():
            total_class_count += self.buffer[label].size(0)
        return total_class_count

    
    def add_data(self, examples, labels):
        """
        Add data to the buffer.
        examples: torch.Size([128, 1, 49, 10])
        labels: torch.Size([128])
        """

        input_size = examples.size(0)
        
        if self.num_seen_examples < self.buffer_size:
            
            for i in range(input_size):
                label = labels[i].item()
                if (label not in self.buffer.keys()) or (self.buffer.items() == {}):
                    # examples: torch.Size([128, 1, 49, 10])
                    # examples[i]: torch.Size([1, 49, 10])
                    self.buffer[label] = torch.unsqueeze(examples[i], 0)
                    # self.buffer[label].append(examples[i])
                    self.class_count[label] = 1
                    self.class_count_total[label] = 1
                else:
                    # self.buffer[label] = [examples[i]]
                    self.buffer[label] = torch.cat((self.buffer[label], torch.unsqueeze(examples[i], 0)), 0)
                    self.class_count[label] += 1
                    self.class_count_total[label] += 1
                self.num_seen_examples += 1
                # print('total_class_count:', self.get_total_class_count())
                # print('class count:', self.get_class_count())
            
        else:
            
            
            for i in range(input_size):
                largest_class_count = max(self.class_count.values())
                largest_classes = set(cls for cls, count in self.class_count.items() if count == largest_class_count)
                # largest_classes_tensor = torch.tensor(list(largest_classes)).to(self.device)

                self.full_classes = list(set(self.full_classes).union(largest_classes))
                # print('full_classes:', self.full_classes)
                label = labels[i].item()
                if label not in self.buffer.keys():
                    # class is new.

                    self.buffer[label] = torch.unsqueeze(examples[i], 0)
                    # randomly select a class from the largest classes
                    random_class = random.choice(list(largest_classes))
                    # random_index = random.randint(0, self.buffer[random_class].size(0) - 1)
                    if self.buffer[random_class].size(0) > 1:
                        random_index = random.randint(0, self.buffer[random_class].size(0) - 1)
                        self.class_count[random_class] -= 1
                    else:
                        random_index = 0
                        self.class_count[random_class] = 0
                    # remove the random_index from the buffer
                    self.buffer[random_class] = torch.cat((self.buffer[random_class][:random_index], self.buffer[random_class][random_index+1:]), 0)
                    
                    self.num_seen_examples += 1
                    # add class count
                    self.class_count[label] = 1
                    self.class_count_total[label] = 1

                elif (label in self.buffer.keys()) and (label not in self.full_classes):
       
                    # class is not in the full classes

                    self.buffer[label] = torch.cat((self.buffer[label], torch.unsqueeze(examples[i], 0)), 0)

                    # TODO: randomly select a class from the largest classes that has the class count larger than 1
                    # Create a list of classes with count larger than 1
                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    # Randomly select a class from the list
                    random_class = random.choice(largest_classes_with_count_larger_than_one) if largest_classes_with_count_larger_than_one else random.choice(list(self.class_count.keys()))
     
                    # random_class = random.choice(list(largest_classes))
                    # random_index = random.randint(0, self.buffer[random_class].size(0) - 1) 
                    if self.buffer[random_class].size(0) > 1:
                        random_index = random.randint(0, self.buffer[random_class].size(0) - 1)
                        self.class_count[random_class] -= 1
                    else:
                        random_index = 0
                        self.class_count[random_class] = 0
                    # remove the random_index from the buffer
                    self.buffer[random_class] = torch.cat((self.buffer[random_class][:random_index], self.buffer[random_class][random_index+1:]), 0)
        
                    self.class_count[label] += 1
                    self.num_seen_examples += 1
                    self.class_count_total[label] += 1
   
                else:
                    # class is in the full classes
                    count_label_current = self.class_count[label]
                    count_label_total = self.class_count_total[label]
                    # sample a u from uniform[0,1]
                    u = int(torch.rand(1))
                    if u <= count_label_current / count_label_total:
        
                        # random_index = random.randint(0, self.buffer[label].size(0) - 1)
                        if self.buffer[label].size(0) > 1:
                            random_index = random.randint(0, self.buffer[label].size(0) - 1)
                            self.buffer[label][random_index] = examples[i]
                        elif self.buffer[label].size(0) == 1:
                            random_index = 0
                            self.buffer[label][random_index] = examples[i]
                        else:
                            random_index = 0
                            self.buffer[label] = torch.unsqueeze(examples[i], 0)
                        # self.buffer[label][random_index] = examples[i]
                        self.num_seen_examples += 1
                        self.class_count_total[label] += 1
                    else:
                        pass
            

    def get_data(self):
        """
        Get data from the buffer.
        """
        # self.buffer[label]
        # label is the key of the dictionary
        # input is the value of the dictionary
        # random select 128 examples from the buffer
        samples = []
        labels = []
        for label, input in self.buffer.items():
            if input.size(0) > 1:
                indices = torch.randperm(input.size(0)).to(self.device)[:1]
                samples.append(input[indices])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                samples.append(input)
                labels.append(torch.tensor([label]).to(self.device))
        return torch.cat(samples, 0), torch.cat(labels, 0)

        # indices = torch.randperm(self.num_seen_examples)[:self.batch_size]
        # return self.buffer['examples'][indices], self.buffer['labels'][indices]
        # indices = torch.randperm(self.buffer_size).to(self.buffer['examples'].device)[:self.batch_size]
        # print(indices)
        # return self.buffer['examples'][indices], self.buffer['labels'][indices]
        
    
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
        return self.class_count
    
    # def get_class_count(self):
    #     """
    #     Get the number of examples for each class in the buffer.
    #     """
    #     class_count = {}
    #     for label in self.buffer['labels']:
    #         if label.item() in class_count:
    #             class_count[label.item()] += 1
    #         else:
    #             class_count[label.item()] = 1
    #     return class_count
