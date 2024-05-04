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
import torchaudio.transforms as T
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from typing import Tuple
from torchvision import transforms
import torch.nn.functional as F

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



# def task_average_forgetting(acc_matrix):

#     acc_final = acc_matrix[-1,:]
#     acc_best = np.max(acc_matrix, axis = 0)
#     forgetting = acc_best - acc_final
#     average_forgetting = np.mean(forgetting)

#     return average_forgetting

def task_average_forgetting(acc_matrix):

    acc_final = acc_matrix[-1,:-1]
    acc_best = np.max(acc_matrix[:,:-1], axis = 0)
    forgetting = acc_best - acc_final
    average_forgetting = np.mean(forgetting)

    return average_forgetting


class RandomAugmentor:
    def __init__(self, device):
        self.device = device
        self.augmentations = [
            "clipping_distortion",
            "time_mask",
            "shift",
            "frequency_mask"
        ]

    def apply(self, examples):
        # Choose a random augmentation
        augmentation = random.choice(self.augmentations)

        # Move tensor to the specified device
        examples = examples.to(self.device)

        if augmentation == "clipping_distortion":
            # Apply clipping distortion
            max_amp = examples.abs().max()
            examples = torch.clamp(examples, min=-max_amp * 0.5, max=max_amp * 0.5)
        
        elif augmentation == "time_mask":
            # Apply time mask
            mask_param = random.randint(1, 10)
            time_mask = T.TimeMasking(time_mask_param=mask_param)
            examples = time_mask(examples)
        
        elif augmentation == "shift":
            # Shift the samples
            shift_amount = random.randint(-5, 5)
            examples = torch.roll(examples, shifts=shift_amount, dims=-1)
        
        elif augmentation == "frequency_mask":
            # Apply frequency mask
            freq_mask_param = random.randint(1, 5)
            freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            examples = freq_mask(examples)
        
        return examples
    

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

    config = config

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
    'seed': config['seed'],
    'volume_ratio': config['volume_ratio'],
    'snr': config['snr'],
    'mode': args.mode,
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
        # if args.mode == 'dil':

        #     if task_id == None:
        #         # not used.
        #         training_parameters['noise_test']  = ['DKITCHEN']

        #         training_parameters['noise_train'] = ['DKITCHEN']

        #     elif task_id == 'dil_task_0':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
        #         training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']

        #     elif task_id == 'dil_task_1':

        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION']

        #         training_parameters['noise_train'] = ['PCAFETER', 'PRESTO', 'PSTATION']

        #     elif task_id == 'dil_task_2':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'SCAFE', 'SPSQUARE', 'STRAFFIC']
        #         training_parameters['noise_train'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC']
                
        #     elif task_id == 'dil_task_3':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'SCAFE', 'SPSQUARE', 'STRAFFIC',\
        #                                             'TBUS', 'TCAR', 'TMETRO']
   
        #         training_parameters['noise_train'] = ['TBUS', 'TCAR', 'TMETRO']

        #     elif task_id == 'dil_joint':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'SCAFE', 'SPSQUARE', 'STRAFFIC',\
        #                                             'TBUS', 'TCAR', 'TMETRO']

        #         training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'SCAFE', 'SPSQUARE', 'STRAFFIC',\
        #                                             'TBUS', 'TCAR', 'TMETRO']
                
        #     if task_id == 'dil_task_0_disjoint':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.
            
        #     elif task_id == 'dil_task_1_disjoint':
        #         training_parameters['noise_test']  = ['PCAFETER', 'PRESTO', 'PSTATION']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.

        #     elif task_id == 'dil_task_2_disjoint':
        #         training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.

        #     elif task_id == 'dil_task_3_disjoint':
        #         training_parameters['noise_test']  = ['TBUS', 'TCAR', 'TMETRO']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.
        
        # if args.mode == 'dil':

        #     if task_id == None:
        #         # not used.
        #         training_parameters['noise_test']  = ['DKITCHEN']

        #         training_parameters['noise_train'] = ['DKITCHEN']

        #     elif task_id == 'dil_task_0':
        #         # training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
        #         # training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']

        #         training_parameters['noise_test'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO']
        #         training_parameters['noise_train'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO']

        #     elif task_id == 'dil_task_1':

        #         training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION']

        #         training_parameters['noise_train'] = ['PCAFETER', 'PRESTO', 'PSTATION']

        #     elif task_id == 'dil_task_2':
        #         training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'DKITCHEN', 'DLIVING', 'DWASHING']
        #         training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING']
                
        #     elif task_id == 'dil_task_3':
        #         training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'DKITCHEN', 'DLIVING', 'DWASHING',\
        #                                             'OHALLWAY', 'OMEETING', 'OOFFICE']
   
        #         training_parameters['noise_train'] = ['OHALLWAY', 'OMEETING', 'OOFFICE']

        #     elif task_id == 'dil_joint':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'SCAFE', 'SPSQUARE', 'STRAFFIC',\
        #                                             'TBUS', 'TCAR', 'TMETRO']

        #         training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE',\
        #                                             'PCAFETER', 'PRESTO', 'PSTATION',\
        #                                             'SCAFE', 'SPSQUARE', 'STRAFFIC',\
        #                                             'TBUS', 'TCAR', 'TMETRO']
                
        #     if task_id == 'dil_task_0_disjoint':
        #         training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.
            
        #     elif task_id == 'dil_task_1_disjoint':
        #         training_parameters['noise_test']  = ['PCAFETER', 'PRESTO', 'PSTATION']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.

        #     elif task_id == 'dil_task_2_disjoint':
        #         training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.

        #     elif task_id == 'dil_task_3_disjoint':
        #         training_parameters['noise_test']  = ['TBUS', 'TCAR', 'TMETRO']
        #         training_parameters['noise_train'] = ['DKITCHEN'] # not used.
        
                
        if args.mode == 'dil':

            if task_id == None:
                # not used.
                training_parameters['noise_test']  = ['DKITCHEN']

                training_parameters['noise_train'] = ['DKITCHEN']

            elif task_id == 'dil_task_0':
                # training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']
                # training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE']

                training_parameters['noise_test'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO']
                training_parameters['noise_train'] = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO']

            elif task_id == 'dil_task_1':

                training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO',\
                                                      'OHALLWAY', 'OMEETING', 'OOFFICE']
                                                    

                training_parameters['noise_train'] = ['OHALLWAY', 'OMEETING', 'OOFFICE']

            elif task_id == 'dil_task_2':
                training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO',\
                                                      'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'DKITCHEN', 'DLIVING', 'DWASHING']
                training_parameters['noise_train'] = ['DKITCHEN', 'DLIVING', 'DWASHING']
                
            elif task_id == 'dil_task_3':
                training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO',\
                                                      'OHALLWAY', 'OMEETING', 'OOFFICE',\
                                                    'DKITCHEN', 'DLIVING', 'DWASHING',\
                                                    'PCAFETER', 'PRESTO', 'PSTATION']
   
                training_parameters['noise_train'] = ['PCAFETER', 'PRESTO', 'PSTATION']

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
                training_parameters['noise_test']  = ['SCAFE', 'SPSQUARE', 'STRAFFIC', 'NFIELD', 'NPARK', 'NRIVER', 'TBUS', 'TCAR', 'TMETRO']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.
            
            elif task_id == 'dil_task_1_disjoint':
                training_parameters['noise_test']  = ['OHALLWAY', 'OMEETING', 'OOFFICE']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.

            elif task_id == 'dil_task_2_disjoint':
                training_parameters['noise_test']  = ['DKITCHEN', 'DLIVING', 'DWASHING']
                training_parameters['noise_train'] = ['DKITCHEN'] # not used.

            elif task_id == 'dil_task_3_disjoint':
                training_parameters['noise_test']  = ['PCAFETER', 'PRESTO', 'PSTATION']
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
    # if args.mode == 'cil':
    #     target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
    # elif args.mode == 'dil':
    #     # 'yes','no','up','down','left','right','on','off','stop','go','backward','bed','bird','cat','dog','eight','five'
    #     target_words = 'yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,'

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


class MiniBatchKMeansTorch:
    def __init__(self, n_clusters=3, n_features=490, max_iter=100, batch_size=10, device='cuda'):
        self.device = torch.device(device)
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.centroids = torch.rand(n_clusters, n_features, device=self.device)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.labels = None

    def fit(self, dataset):
        dataset = dataset.clone().detach().to(self.device)
        self.labels = torch.empty(dataset.size(0), dtype=torch.long, device=self.device)
        for _ in range(self.max_iter):
            minibatch_indices = torch.randint(0, dataset.size(0), (self.batch_size,), device=self.device)
            minibatch = dataset[minibatch_indices]
            distances = torch.cdist(minibatch, self.centroids, p=2)
            labels = torch.argmin(distances, dim=1)
            self.labels[minibatch_indices] = labels
            for i in range(self.n_clusters):
                cluster_points = minibatch[labels == i]
                if cluster_points.size(0) > 0:
                    self.centroids[i] = cluster_points.mean(dim=0)

    def partial_fit(self, dataset, new_data):
        # First ensure that labels are aligned with the dataset
        if self.labels is None or len(self.labels) != len(dataset):
            # Reinitialize labels if they are not aligned
            self.labels = torch.full((len(dataset),), -1, dtype=torch.long, device=dataset.device)

        # Proceed with existing logic
        distances_to_centroids = torch.cdist(new_data, self.centroids, p=2)
        closest_cluster = torch.argmin(distances_to_centroids, dim=1).item()
        indices_in_cluster = (self.labels == closest_cluster).nonzero(as_tuple=True)[0]
        
        if indices_in_cluster.size(0) > 0:
            random_idx_to_replace = indices_in_cluster[torch.randint(0, indices_in_cluster.size(0), (1,))].item()
            dataset[random_idx_to_replace] = new_data.squeeze()
            # print(f"Replaced index: {random_idx_to_replace} in cluster {closest_cluster}")
            affected_data = dataset[self.labels == closest_cluster]
            self.centroids[closest_cluster] = affected_data.mean(dim=0)

        return dataset

    def remove_sample_from_largest_cluster(self, dataset):
        # if self.labels is None or len(self.labels) != len(dataset):
        #     raise ValueError("Dataset and labels are not aligned or labels are uninitialized.")

        # Count the number of samples in each cluster
        cluster_counts = torch.bincount(self.labels, minlength=self.n_clusters)

        # Find the cluster with the maximum count
        largest_cluster = torch.argmax(cluster_counts).item()

        # Get indices of all samples in the largest cluster
        indices_in_largest_cluster = (self.labels == largest_cluster).nonzero(as_tuple=True)[0]

        if indices_in_largest_cluster.size(0) > 0:
            # Randomly select one sample to remove
            idx_to_remove = indices_in_largest_cluster[torch.randint(0, indices_in_largest_cluster.size(0), (1,))].item()
            # Remove the sample by replacing it with the last sample and shrinking the dataset
            dataset[idx_to_remove] = dataset[-1]
            dataset = dataset[:-1]
            self.labels[idx_to_remove] = self.labels[-1]
            self.labels = self.labels[:-1]

            # print(f"Removed index: {idx_to_remove} from cluster {largest_cluster}, New dataset size: {len(dataset)}")

        return dataset

    def predict(self, X):
        X = X.to(self.device)
        distances = torch.cdist(X, self.centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        return labels.cpu().numpy()


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


class Buffer_ECB:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.num_seen_examples = 0
        self.buffer = {}

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

                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    sorted_class_count_total = {k: v for k, v in sorted(self.class_count_total.items(), key=lambda item: item[1], reverse=True)}

                    random_class = random.choice(list(largest_classes_with_count_larger_than_one))

                    exp_n = np.exp(-self.class_count_total[label])
                    exp_sum = np.sum([np.exp(-v) for v in sorted_class_count_total.values()])
                    w = exp_n / exp_sum
                    gamma = self.buffer_size * w
                    
                    for key in sorted_class_count_total:
                        if key in largest_classes_with_count_larger_than_one and self.class_count[key] >= gamma:
                            random_class = key
                            break

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
            

    def get_data(self, input_size):
        """
        Get data from the buffer.
        """

        samples = []
        labels = []

        for i in range(input_size):
            label = random.choice(list(self.buffer.keys()))
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                samples.append(self.buffer[label])
                labels.append(torch.tensor([label]).to(self.device))

        return torch.cat(samples, 0), torch.cat(labels, 0)
        
    
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
    

class Buffer_DAECB:
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

                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    # order the self.class_count_total.keys() by the value of self.class_count_total
                    sorted_class_count_total = {k: v for k, v in sorted(self.class_count_total.items(), key=lambda item: item[1], reverse=True)}
                    # # take intersection of largest_classes_with_count_larger_than_one and sorted_class_count_total
                    # largest_classes_with_count_larger_than_one = list(set(largest_classes_with_count_larger_than_one).intersection(set(sorted_class_count_total)))
                    random_class = random.choice(list(largest_classes_with_count_larger_than_one))

                    exp_n = np.exp(-self.class_count_total[label])
                    exp_sum = np.sum([np.exp(-v) for v in sorted_class_count_total.values()])
                    w = exp_n / exp_sum
                    gamma = self.buffer_size * w
                    
                    for key in sorted_class_count_total:
                        if key in largest_classes_with_count_larger_than_one and self.class_count[key] >= gamma:
                            random_class = key
                            break
                        
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
                        
                        # calculate cosine similarity between the current example and the examples in the buffer
                        examples_in_buffer = self.buffer[label] # torch.Size([n, 49, 10])
                        # flatten the examples_in_buffer
                        examples_in_buffer = examples_in_buffer.view(examples_in_buffer.size(0), -1)
                        current_example = examples[i] # torch.Size([1, 49, 10])
                        # flatten the current_example
                        current_example = current_example.view(1, -1)
                        # print('examples_in_buffer:', examples_in_buffer.size()) # torch.Size([n, 490])
                        # print('current_example:', current_example.size()) # torch.Size([1, 490])

                        # Calculate cosine similarity
                        cos_similarities = F.cosine_similarity(examples_in_buffer.to(self.device), current_example.to(self.device), dim=-1) # torch.Size([n])
                        # normalize the cosine similarities[-1,1]
                        cos_similarities = 0.5 * (cos_similarities + 1)
                        # randomly select an example from the buffer according to the cosine similarity
                        sample_index = torch.multinomial(cos_similarities, 1).item()

                        # # Calculate Euclidean distance
                        # euclidean_distances = torch.norm(examples_in_buffer.to(self.device) - current_example.to(self.device), dim=-1)
                        # # randomly select an example from the buffer according to the euclidean_distances
                        # sample_index = torch.multinomial(euclidean_distances, 1).item()

                        # sample_index = torch.argmax(cos_similarities)
                        self.buffer[label][sample_index] = examples[i]

                        self.num_seen_examples += 1
                        self.class_count_total[label] += 1
                    else:
                        pass
            

    def get_data(self, input_size):
        """
        Get data from the buffer.
        """

        samples = []
        labels = []

        for i in range(input_size):
            label = random.choice(list(self.buffer.keys()))
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                samples.append(self.buffer[label])
                labels.append(torch.tensor([label]).to(self.device))

        return torch.cat(samples, 0), torch.cat(labels, 0)
        
    
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


class Buffer_ECB_custom:
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

                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    # order the self.class_count_total.keys() by the value of self.class_count_total
                    sorted_class_count_total = {k: v for k, v in sorted(self.class_count_total.items(), key=lambda item: item[1], reverse=True)}
                    # # take intersection of largest_classes_with_count_larger_than_one and sorted_class_count_total
                    # largest_classes_with_count_larger_than_one = list(set(largest_classes_with_count_larger_than_one).intersection(set(sorted_class_count_total)))
                    random_class = random.choice(list(largest_classes_with_count_larger_than_one))

                    exp_n = np.exp(-self.class_count_total[label])
                    exp_sum = np.sum([np.exp(-v) for v in sorted_class_count_total.values()])
                    w = exp_n / exp_sum
                    gamma = self.buffer_size * w
                    
                    for key in sorted_class_count_total:
                        if key in largest_classes_with_count_larger_than_one and self.class_count[key] >= gamma:
                            random_class = key
                            break
                        
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
                        
                        # calculate cosine similarity between the current example and the examples in the buffer
                        examples_in_buffer = self.buffer[label] # torch.Size([n, 49, 10])
                        # flatten the examples_in_buffer
                        examples_in_buffer = examples_in_buffer.view(examples_in_buffer.size(0), -1)
                        current_example = examples[i] # torch.Size([1, 49, 10])
                        # flatten the current_example
                        current_example = current_example.view(1, -1)
                        # print('examples_in_buffer:', examples_in_buffer.size()) # torch.Size([n, 490])
                        # print('current_example:', current_example.size()) # torch.Size([1, 490])

                        # Calculate cosine similarity
                        cos_similarities = F.cosine_similarity(examples_in_buffer.to(self.device), current_example.to(self.device), dim=-1) # torch.Size([n])
                        # normalize the cosine similarities[-1,1]
                        cos_similarities = 0.5 * (cos_similarities + 1)
                        # randomly select an example from the buffer according to the cosine similarity
                        sample_index = torch.multinomial(cos_similarities, 1).item()

                        # # Calculate Euclidean distance
                        # euclidean_distances = torch.norm(examples_in_buffer.to(self.device) - current_example.to(self.device), dim=-1)
                        # # randomly select an example from the buffer according to the euclidean_distances
                        # sample_index = torch.multinomial(euclidean_distances, 1).item()

                        # sample_index = torch.argmax(cos_similarities)
                        self.buffer[label][sample_index] = examples[i]

                        self.num_seen_examples += 1
                        self.class_count_total[label] += 1
                    else:
                        pass
            

    def get_data(self, input_size):
        """
        Get data from the buffer.
        """

        samples = []
        labels = []

        for i in range(input_size):
            label = random.choice(list(self.buffer.keys()))
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                samples.append(self.buffer[label])
                labels.append(torch.tensor([label]).to(self.device))

        return torch.cat(samples, 0), torch.cat(labels, 0)
        
    
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



class Buffer_LossECB:
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

        self.loss_per_class = None
        print("Buffer initialized")


    def get_total_class_count(self):
        """
        Get the total number of examples for each class in the self.buffer[label]
        """
        total_class_count = 0
        for label in self.buffer.keys():
            total_class_count += self.buffer[label].size(0)
        return total_class_count

    
    def add_data(self, examples, labels, loss_per_class):
        """
        Add data to the buffer.
        examples: torch.Size([128, 1, 49, 10])
        labels: torch.Size([128])
        max_loss_class: the classes and their losses
        """
        self.loss_per_class = loss_per_class # dict. keys are the classes and values are the mean losses.
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

                    # ECB:
                    # largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    # # order the self.class_count_total.keys() by the value of self.class_count_total
                    # sorted_class_count_total = {k: v for k, v in sorted(self.class_count_total.items(), key=lambda item: item[1], reverse=True)}
                    # # # take intersection of largest_classes_with_count_larger_than_one and sorted_class_count_total
                    # # largest_classes_with_count_larger_than_one = list(set(largest_classes_with_count_larger_than_one).intersection(set(sorted_class_count_total)))
                    # random_class = random.choice(list(largest_classes_with_count_larger_than_one))

                    # exp_n = np.exp(-self.class_count_total[label])
                    # exp_sum = np.sum([np.exp(-v) for v in sorted_class_count_total.values()])
                    # w = exp_n / exp_sum
                    # gamma = self.buffer_size * w
                    
                    # for key in sorted_class_count_total:
                    #     if key in largest_classes_with_count_larger_than_one and self.class_count[key] >= gamma:
                    #         random_class = key
                    #         break

                    # LossECB:
                    # largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]
                    # # randomly select a class from largest_classes_with_count_larger_than_one according to the loss
                    # random_class = torch.multinomial(torch.tensor([1/self.loss_per_class[class_label] for class_label in largest_classes_with_count_larger_than_one]), 1).item()
                    # random_class = largest_classes_with_count_larger_than_one[random_class]
                    # print('random_class:', random_class)
                    # Filter out classes that are not in self.loss_per_class
                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]
                    valid_classes = [class_label for class_label in largest_classes_with_count_larger_than_one if class_label in self.loss_per_class]

                    # Calculate the probabilities
                    probabilities = [1/self.loss_per_class[class_label] for class_label in valid_classes]

                    # Check if probabilities list is empty
                    if not probabilities:
                        raise ValueError("No valid classes to select from")

                    # Select a random class
                    random_class = torch.multinomial(torch.tensor(probabilities), 1).item()
                    random_class = valid_classes[random_class]
                    
                        
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
            

    def get_data(self, input_size):
        """
        Get data from the buffer.
        """

        samples = []
        labels = []

        for i in range(input_size):
            label = random.choice(list(self.buffer.keys()))
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                samples.append(self.buffer[label])
                labels.append(torch.tensor([label]).to(self.device))

        return torch.cat(samples, 0), torch.cat(labels, 0)
        
    
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
    
        
class Buffer_ClusterECB:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, batch_size, config, device):
        self.config = config
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

        self.old_classes = None

        self.fit_flags = {}

        print("Buffer initialized")

    def update_old_classes(self):
        self.old_classes = self.buffer.keys()
    
    def get_old_classes(self):
        return self.old_classes
    
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
                    # class is new.

                    # examples: torch.Size([128, 1, 49, 10])
                    # examples[i]: torch.Size([1, 49, 10])
                    self.buffer[label] = torch.unsqueeze(examples[i], 0)
                    # self.buffer[label].append(examples[i])
                    # init the cluster for the new class
                    self.clusters[label] = MiniBatchKMeansTorch(n_clusters=self.config['n_clusters'], n_features=490, max_iter=100, batch_size=10, device=self.device)
                    self.fit_flags[label] = True
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

                if label not in self.buffer:
                    # class is new.

                    self.buffer[label] = torch.unsqueeze(examples[i], 0) 
                    # init the cluster for the new class
                    self.clusters[label] = MiniBatchKMeansTorch(n_clusters=self.config['n_clusters'], n_features=490, max_iter=100, batch_size=10, device=self.device)
                    self.fit_flags[label] = True
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

                elif (label in self.buffer) and (label not in self.full_classes):
       
                    # class is not in the full classes

                    self.buffer[label] = torch.cat((self.buffer[label], torch.unsqueeze(examples[i], 0)), 0) # 

                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    # order the self.class_count_total.keys() by the value of self.class_count_total
                    sorted_class_count_total = {k: v for k, v in sorted(self.class_count_total.items(), key=lambda item: item[1], reverse=True)}
                    # # take intersection of largest_classes_with_count_larger_than_one and sorted_class_count_total
                    # largest_classes_with_count_larger_than_one = list(set(largest_classes_with_count_larger_than_one).intersection(set(sorted_class_count_total)))
                    random_class = random.choice(list(largest_classes_with_count_larger_than_one))

                    exp_n = np.exp(-self.class_count_total[label])
                    exp_sum = np.sum([np.exp(-v) for v in sorted_class_count_total.values()])
                    w = exp_n / exp_sum
                    gamma = self.buffer_size * w
                    
                    for key in sorted_class_count_total:
                        if key in largest_classes_with_count_larger_than_one and self.class_count[key] >= gamma:
                            random_class = key
                            break
                        
                    if self.buffer[random_class].size(0) > 1:
                        random_index = random.randint(0, self.buffer[random_class].size(0) - 1)
                        self.class_count[random_class] -= 1
                    else:
                        random_index = 0
                        self.class_count[random_class] = 0
                    # remove the random_index from the buffer

                    # examples_in_buffer = self.buffer[random_class] # torch.Size([n, 49, 10])
                    # # flatten the examples_in_buffer
                    # examples_in_buffer = examples_in_buffer.view(examples_in_buffer.size(0), -1)
                    # current_example = examples[i] # torch.Size([1, 49, 10])
                    # # flatten the current_example
                    # current_example = current_example.view(1, -1)

                    # self.clusters[random_class].fit(examples_in_buffer.to(self.device))

                    # examples_in_buffer = self.clusters[random_class].remove_sample_from_largest_cluster(examples_in_buffer)

                    # self.buffer[random_class] = examples_in_buffer.view(examples_in_buffer.size(0), 1, 49, 10)


                    self.buffer[random_class] = torch.cat((self.buffer[random_class][:random_index], self.buffer[random_class][random_index+1:]), 0)
        
                    self.class_count[label] += 1
                    self.num_seen_examples += 1
                    self.class_count_total[label] += 1
   
                else:
                    
                    # class is in the full classes
                    count_label_current = self.class_count[label]
                    count_label_total = self.class_count_total[label]
                    # print('len',len(self.buffer.keys()))
                    # print('fit_flags', self.fit_flags)
                    if (count_label_current >= int((self.buffer_size / len(self.buffer.keys()))-2)) and (self.fit_flags[label] == True):
                        examples_in_buffer = self.buffer[label] # torch.Size([n, 49, 10])
                        # flatten the examples_in_buffer
                        examples_in_buffer = examples_in_buffer.view(examples_in_buffer.size(0), -1)
                        self.clusters[label].fit(examples_in_buffer.to(self.device))
                        self.fit_flags[label] = False
                    
                    # sample a u from uniform[0,1]
                    u = int(torch.rand(1))
                    if u <= count_label_current / count_label_total:
                        

                        # calculate cosine similarity between the current example and the examples in the buffer
                        examples_in_buffer = self.buffer[label] # torch.Size([n, 49, 10])
                        # flatten the examples_in_buffer
                        examples_in_buffer = examples_in_buffer.view(examples_in_buffer.size(0), -1)
                        current_example = examples[i] # torch.Size([1, 49, 10])
                        # flatten the current_example
                        current_example = current_example.view(1, -1)
                        # print('examples_in_buffer:', examples_in_buffer.size()) # torch.Size([n, 490])
                        # print('current_example:', current_example.size()) # torch.Size([1, 490])
                        # if count_label_current is larger than buffer_size/number of keys in self.buffer
                        if self.fit_flags[label] is True:
                            self.clusters[label].fit(examples_in_buffer)
                            self.fit_flags[label] = False
                        else:
                            examples_in_buffer = self.clusters[label].partial_fit(examples_in_buffer, current_example)
     
                        # self.buffer[label]: torch.Size([n, 49, 10])
                        self.buffer[label] = examples_in_buffer.view(examples_in_buffer.size(0), 1, 49, 10)


                        # sample_index = 
                        # self.buffer[label][sample_index] = examples[i]

                        self.num_seen_examples += 1
                        self.class_count_total[label] += 1
                    else:
                        pass
            

    def get_data(self, input_size):
        """
        Get data from the buffer.
        """

        samples = []
        labels = []

        for i in range(input_size):
            label = random.choice(list(self.buffer.keys()))
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                pass
                # samples.append(self.buffer[label])
                # labels.append(torch.tensor([label]).to(self.device))

        return torch.cat(samples, 0), torch.cat(labels, 0)
    
    
        
    
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
   

class Buffer_LAECB:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.num_seen_examples = 0
        self.buffer = {}
        self.loss = {}

        self.loss_index = {}

        self.class_count = {}

        self.class_count_total = {}

        self.full_classes = []
        print("Buffer initialized")
    
    def load_buffer(self, buffer_state):
        '''
        load the buffer
        buffer_state = {'num_seen_examples': num_seen_examples, 
                        'buffer': buffer, 
                        'loss': loss, 
                        'loss_index': loss_index, 
                        'class_count': class_count, 
                        'class_count_total': class_count_total, 
                        'full_classes': full_classes}
        '''
        self.num_seen_examples = buffer_state['num_seen_examples']
        self.buffer = buffer_state['buffer']
        self.loss = buffer_state['loss']
        self.loss_index = buffer_state['loss_index']
        self.class_count = buffer_state['class_count']
        self.class_count_total = buffer_state['class_count_total']
        self.full_classes = buffer_state['full_classes']
        

    def get_entire_buffer(self):
        '''
        get everything in the buffer
        self.num_seen_examples: int
        self.buffer: dict
        self.loss: dict
        self.loss_index: dict
        self.class_count: dict
        self.class_count_total: dict
        self.full_classes: list
        '''
        return self.num_seen_examples, self.buffer, self.loss, self.loss_index, self.class_count, self.class_count_total, self.full_classes

    def get_total_class_count(self):
        """
        Get the total number of examples for each class in the self.buffer[label]
        """
        total_class_count = 0
        for label in self.buffer.keys():
            total_class_count += self.buffer[label].size(0)
        return total_class_count
    
    def update_old_classes(self):
        self.old_classes = self.buffer.keys()
    
    def get_old_classes(self):
        return self.old_classes

    
    def add_data(self, examples, losses, labels):
        """
        Add data to the buffer.
        examples: torch.Size([128, 1, 49, 10])
        losses: torch.Size([128])
        labels: torch.Size([128])
        """
        # make losses not updated
        # losses = losses.detach()
        input_size = examples.size(0)
        
        if self.num_seen_examples < self.buffer_size:
            
            for i in range(input_size):
                label = labels[i].item()
                if (label not in self.buffer.keys()) or (self.buffer.items() == {}):
                    # examples: torch.Size([128, 1, 49, 10])
                    # examples[i]: torch.Size([1, 49, 10])
                    # loss[i]: torch.Size([1])
                    self.buffer[label] = torch.unsqueeze(examples[i], 0)
                    self.loss[label] = torch.unsqueeze(losses[i], 0) # torch.Size([1])
                    # self.loss[label] = torch.unsqueeze(losses[i], 0)
                    # self.buffer[label].append(examples[i])
                    self.class_count[label] = 1
                    self.class_count_total[label] = 1
                else:
                    # self.buffer[label] = [examples[i]]
                    self.buffer[label] = torch.cat((self.buffer[label], torch.unsqueeze(examples[i], 0)), 0)

                    self.loss[label] = torch.cat((self.loss[label], torch.unsqueeze(losses[i], 0)), 0) # torch.Size([n])
                    # self.loss[label] = torch.cat((self.loss[label], torch.unsqueeze(losses[i], 0)), 0)
                    self.class_count[label] += 1
                    self.class_count_total[label] += 1
                self.num_seen_examples += 1
            
        else:
            # print('self.loss:', self.loss)
            # print('self.buffer.keys()', self.buffer.keys())
            for i in range(input_size):
                largest_class_count = max(self.class_count.values())
                largest_classes = set(cls for cls, count in self.class_count.items() if count == largest_class_count)

                self.full_classes = list(set(self.full_classes).union(largest_classes))

                label = labels[i].item()
                if label not in self.buffer.keys():
                    # class is new.

                    self.buffer[label] = torch.unsqueeze(examples[i], 0)
                    self.loss[label] = torch.unsqueeze(losses[i], 0)
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
                    self.loss[random_class] = torch.cat((self.loss[random_class][:random_index], self.loss[random_class][random_index+1:]), 0)
                    # self.loss[random_class] = torch.cat((self.loss[random_class][:random_index], self.loss[random_class][random_index+1:]), 0)

                    self.num_seen_examples += 1
                    # add class count
                    self.class_count[label] = 1
                    self.class_count_total[label] = 1

                elif (label in self.buffer.keys()) and (label not in self.full_classes):
       
                    # class is not in the full classes

                    self.buffer[label] = torch.cat((self.buffer[label], torch.unsqueeze(examples[i], 0)), 0)
     
                    self.loss[label] = torch.cat((self.loss[label], torch.unsqueeze(losses[i], 0)), 0)
                    # self.loss[label] = torch.cat((self.loss[label], torch.unsqueeze(losses[i], 0)), 0)

                    largest_classes_with_count_larger_than_one = [class_label for class_label in largest_classes if self.class_count[class_label] > 1]

                    # order the self.class_count_total.keys() by the value of self.class_count_total
                    sorted_class_count_total = {k: v for k, v in sorted(self.class_count_total.items(), key=lambda item: item[1], reverse=True)}
                    # # take intersection of largest_classes_with_count_larger_than_one and sorted_class_count_total
                    # largest_classes_with_count_larger_than_one = list(set(largest_classes_with_count_larger_than_one).intersection(set(sorted_class_count_total)))
                    random_class = random.choice(list(largest_classes_with_count_larger_than_one))

                    exp_n = np.exp(-self.class_count_total[label])
                    exp_sum = np.sum([np.exp(-v) for v in sorted_class_count_total.values()])
                    w = exp_n / exp_sum
                    gamma = self.buffer_size * w
                    
                    for key in sorted_class_count_total:
                        if key in largest_classes_with_count_larger_than_one and self.class_count[key] >= gamma:
                            random_class = key
                            break
                        
                    if self.buffer[random_class].size(0) > 1:
                        random_index = random.randint(0, self.buffer[random_class].size(0) - 1)
                        self.class_count[random_class] -= 1
                    else:
                        random_index = 0
                        self.class_count[random_class] = 0
                    # remove the random_index from the buffer
                    self.buffer[random_class] = torch.cat((self.buffer[random_class][:random_index], self.buffer[random_class][random_index+1:]), 0)
        
                    self.loss[random_class] = torch.cat((self.loss[random_class][:random_index], self.loss[random_class][random_index+1:]), 0)
                    # self.loss[random_class] = torch.cat((self.loss[random_class][:random_index], self.loss[random_class][random_index+1:]), 0)

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
                            # randomly select an example from the buffer according to the loss distribution
                            
                            random_index = torch.multinomial(F.softmax(-self.loss[label], dim=0), 1).item()
                            self.buffer[label][random_index] = examples[i]
                            self.loss[label][random_index] = losses[i]

                                    
                        elif self.buffer[label].size(0) == 1:
                            random_index = 0
                            self.buffer[label][random_index] = examples[i]
                            self.loss[label][random_index] = losses[i]
                        else:
                            random_index = 0
                            self.buffer[label] = torch.unsqueeze(examples[i], 0)
                            self.loss[label] = torch.unsqueeze(losses[i], 0)
                        # self.buffer[label][random_index] = examples[i]
                        self.num_seen_examples += 1
                        self.class_count_total[label] += 1

                        
                    else:
                        pass
            

    def get_data(self, input_size):
        """
        Get data from the buffer.
        """
        # self.buffer[label]
        # label is the key of the dictionary
        # input is the value of the dictionary
        # random select 128 examples from the buffer

        # self.loss_index is dict and is initialized with keys of self.buffer.keys()
        self.loss_index = {key: [] for key in self.buffer.keys()}
        
        samples = []
        # losses = [] # record the index of where the loss is sampled
        labels = []
        for i in range(input_size):
            
            label = random.choice(list(self.buffer.keys()))
            # print('label:', label)
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                # losses.append(self.loss[label][indices])
                self.loss_index[label].append(indices)
                # print('self.loss_index[label]:', self.loss_index[label])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                pass

        return torch.cat(samples, 0), torch.cat(labels, 0)
    
    def get_new_data(self):
        """
        Get data from the buffer.
        """
        # self.buffer[label]
        # label is the key of the dictionary
        # input is the value of the dictionary
        # random select 128 examples from the buffer

        # self.loss_index is dict and is initialized with keys of self.buffer.keys()
        self.loss_index = {key: [] for key in self.buffer.keys()}
        
        # get the classes that is in self.buffer.keys() but not in self.old_classes
        new_classes = list(set(self.buffer.keys()).difference(set(self.old_classes)))
        # get total number of samples in the new classes
        total_samples = 0
        for label in new_classes:
            total_samples += self.buffer[label].size(0)

        samples = []
        # losses = [] # record the index of where the loss is sampled
        labels = []

        for i in range(total_samples):
            
            # label = random.choice(list(self.buffer.keys()))
            label = random.choice(new_classes)
            # print('label:', label)
            if self.buffer[label].size(0) >= 1:
                indices = torch.randperm(self.buffer[label].size(0)).to(self.device)[:1]
                samples.append(self.buffer[label][indices])
                # losses.append(self.loss[label][indices])
                self.loss_index[label].append(indices)
                # print('self.loss_index[label]:', self.loss_index[label])
                labels.append(torch.tensor([label]).to(self.device))
            else:
                pass

        return torch.cat(samples, 0), torch.cat(labels, 0)
    
    
    def update_loss(self, losses, labels):
        """
        Update the loss of the examples in the buffer according to self.loss_index
        losses: torch.Size([input_size])
        """
        
        # print('self.loss_index.keys()', self.loss_index.keys())
        # print('self.loss.keys()', self.loss)

        for i in range(losses.size(0)):
            label = labels[i].item()
            
            for j in range(len(self.loss_index[label])):
                if label not in self.loss.keys():
                    self.loss[label] = torch.unsqueeze(losses[i], 0)
                else:
                    self.loss[label][self.loss_index[label][j]] = losses[i]
        


        # print(self.loss_index.keys())
        # for label in self.loss_index.keys():
        #     # print('losses', losses.shape)
        #     # print('self.loss[label]',self.loss[label].shape, 'self.buffer[label].shape', self.buffer[label].shape)
        #     # print(f'{label}:',list(self.loss_index[label]))

        #     for i in range(len(self.loss_index[label])):
        #         if label not in self.loss.keys():
                    
        #             self.loss[label] = torch.unsqueeze(losses[i], 0)
        #             print('losses:', self.loss[label])
        #         else:
        #             self.loss[label][self.loss_index[label][i]] = losses[i]
                

    
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
