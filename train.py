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

import time
import dataset
import dataset_copy
from sklearn.metrics import confusion_matrix
import torch
import wandb
from typing import Optional
from utils import confusion_matrix, npy_to_txt, per_noise_accuracy, load_config, RandomAugmentor, \
                augment_mfcc
import copy
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class Train():

    def __init__(self, audio_processor, training_parameters, model, device, args, config):
        self.audio_processor = audio_processor
        self.training_parameters = training_parameters
        self.model = model
        self.device = device
        self.args = args
        self.config = config
        torch.manual_seed(self.config['seed'])
        # Training hyperparameters
        self.criterion = torch.nn.CrossEntropyLoss()
        intitial_lr = 0.001
        if not self.args.reg:
            
            self.optimizer = torch.optim.Adam(model.parameters(), lr = intitial_lr)
        else:

            self.optimizer = torch.optim.SGD(model.parameters(), lr=intitial_lr, weight_decay=1e-4)

        lambda_lr = lambda epoch: 1 if epoch<15 else 1/5 if epoch < 25 else 1/10 if epoch<35 else 1/20
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)
    
    def criterion_kd(self, old_logits, new_logits, temperature=2.0):
        """
        Compute the knowledge distillation loss between the old and new logits.
        
        Args:
            old_logits (torch.Tensor): Logits from the model before training.
            new_logits (torch.Tensor): Logits from the model after training.
            temperature (float): Temperature scaling factor.
        
        Returns:
            torch.Tensor: The knowledge distillation loss.
        """
        # Apply temperature scaling
        old_logits = old_logits / temperature
        new_logits = new_logits / temperature
        
        # Compute softmax and log-softmax
        softmax_old_logits = torch.nn.functional.softmax(old_logits, dim=1)
        log_softmax_new_logits = torch.nn.functional.log_softmax(new_logits, dim=1)
        
        # Compute the distillation loss
        loss = -torch.sum(softmax_old_logits * log_softmax_new_logits, dim=1).mean()
        return loss

    # Mixup function
    def mixup_data(x, y):
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        alpha=1.0
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def validate(self, model = None, mode='validation', batch_size = 0, task_id = None,
                 statistics = False, integer = False, save = False, task = None):
        # Validate model

        training_parameters = self.training_parameters
        if (batch_size != 0):
            training_parameters['batch_size'] = batch_size  

        data = dataset.AudioGenerator(mode, self.audio_processor, training_parameters, task_id, task = task)
        model.eval()  

        correct = 0
        total = 0

        with torch.no_grad():

            if batch_size == -1:
                
                inputs, labels, noises = data[0]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                # labels = torch.Tensor(labels).long().to(self.device)
                inputs.unsqueeze_(1)
                model = model.to(self.device)  
                noise_types = list(set(noises))
                # print(noise_types)

                if (integer):
                    model = model.cpu()
                    inputs = inputs * 255./255 
                    inputs = inputs.type(torch.uint8).type(torch.float).cpu()           

                if (save):
                    model = model.cpu()
                    inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                    outputs = F.softmax(model(inputs, save), dim=1)
                    outputs = outputs.to(self.device)
                    npy_to_txt(-1, inputs.int().cpu().detach().numpy())
                else:
                    outputs = F.softmax(model(inputs), dim=1)
                    outputs = outputs.to(self.device)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if statistics == True:
                    # conf_matrix(labels, predicted, self.training_parameters)
                    per_noise_accuracy(labels, predicted, noises)

            else:
                # print(mode)
                num_iter = len(data)
                if task_id == 'cil_task_0':
                    num_iter = int((17/35)*num_iter)
                elif task_id == 'cil_task_1':
                    num_iter = int((23/35)*num_iter)
                elif task_id == 'cil_task_2':
                    num_iter = int((29/35)*num_iter)
                elif task_id == 'cil_task_3':
                    num_iter = int((35/35)*num_iter)
                # print(num_iter)
                for minibatch in range(num_iter):
                    inputs, labels, noises = data[0]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).long().to(self.device)
                    inputs.unsqueeze_(1)
                    model = model.to(self.device)  

                    if (integer):
                        model = model.cpu()
                        inputs = inputs * 255./255 
                        inputs = inputs.type(torch.uint8).type(torch.float).cpu()           

                    if (save):
                        model = model.cpu()
                        inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                        outputs = F.softmax(model(inputs, save), dim=1)
                        outputs = outputs.to(self.device)
                        npy_to_txt(-1, inputs.int().cpu().detach().numpy())
                    else:
                        outputs = F.softmax(model(inputs), dim=1)
                        outputs = outputs.to(self.device)

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # calculate F1 score

                    if minibatch % 20 == 0: 
                        print('[%3d / %3d] accuracy: %.3f' % (minibatch + 1, num_iter,  100 * correct / total))
                        
                        running_loss = 0.0

                    if statistics == True:
                        # conf_matrix(labels, predicted, self.training_parameters)
                        per_noise_accuracy(labels, predicted, noises)
        if mode == 'validation':
            print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        elif mode == 'testing':
            pass
        if self.args.wandb and mode == 'validation':
            wandb.log({'val_accuracy': 100 * correct / total})
        elif self.args.wandb and mode == 'testing':
            wandb.log({'test_accuracy': 100 * correct / total})

        return(100 * correct / total)


    def validate_noise(self, model = None, mode='validation', batch_size = 0, task_id = None,
                 statistics = False, integer = False, save = False, task = None):
        # Validate model

        noise_types = ['TBUS_48k', 'OMEETING_48k', 'OHALLWAY_48k', 'NFIELD_48k', 'NPARK_48k', 'NRIVER_48k', \
                       'OOFFICE_48k', 'SPSQUARE_48k', 'DLIVING_48k', 'TCAR_48k', 'STRAFFIC_48k', 'PSTATION_48k', \
                       'DKITCHEN_48k', 'TMETRO_48k', 'DWASHING_48k', 'PCAFETER_48k', 'PRESTO_48k', 'SCAFE_48k' ]
        
        # TBUS_48k: 0, OMEETING_48k: 1, OHALLWAY_48k: 2, NFIELD_48k: 3, NPARK_48k: 4, NRIVER_48k: 5, ...
        noise_dict = {noise_types[i]: i for i in range(len(noise_types))}
        print(noise_dict)

        training_parameters = self.training_parameters
        if (batch_size != 0):
            training_parameters['batch_size'] = batch_size  

        data = dataset_copy.AudioGenerator(mode, self.audio_processor, training_parameters, task_id, task = task)
        model.eval()  

        correct = 0
        total = 0

        with torch.no_grad():

            if batch_size == -1:
                
                inputs, labels, noises = data[0]
                inputs = inputs.to(self.device)

                noises = [noise_dict[noise] for noise in noises]
                noises = torch.Tensor(noises).to(self.device).long()

                inputs.unsqueeze_(1)
                model = model.to(self.device)  



                if (integer):
                    model = model.cpu()
                    inputs = inputs * 255./255 
                    inputs = inputs.type(torch.uint8).type(torch.float).cpu()           

                if (save):
                    model = model.cpu()
                    inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                    outputs = F.softmax(model(inputs, save), dim=1)
                    outputs = outputs.to(self.device)
                    npy_to_txt(-1, inputs.int().cpu().detach().numpy())
                else:
                    outputs = F.softmax(model(inputs), dim=1)
                    outputs = outputs.to(self.device)

                _, predicted = torch.max(outputs, 1)
                total += noises.size(0)
                correct += (predicted == noises).sum().item()

                # Move predictions and labels to CPU for confusion matrix calculation
                predicted = predicted.cpu().numpy()
                noises = noises.cpu().numpy()

                # Calculate confusion matrix
                cm = confusion_matrix(noises, predicted)

                # Print confusion matrix
                print("Confusion Matrix:")
                print(cm)

                # print accuracy
                print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))



            else:
                # print(mode)
                num_iter = len(data)
                if task_id == 'cil_task_0':
                    num_iter = int((17/35)*num_iter)
                elif task_id == 'cil_task_1':
                    num_iter = int((23/35)*num_iter)
                elif task_id == 'cil_task_2':
                    num_iter = int((29/35)*num_iter)
                elif task_id == 'cil_task_3':
                    num_iter = int((35/35)*num_iter)
                # print(num_iter)
                for minibatch in range(num_iter):
                    inputs, labels, noises = data[0]
                    inputs = inputs.to(self.device)
                    # labels = labels.to(self.device)

                    noises = [noise_dict[noise] for noise in noises]
                    noises = torch.Tensor(noises).to(self.device).long()

                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).long().to(self.device)
                    inputs.unsqueeze_(1)
                    model = model.to(self.device)  

                    if (integer):
                        model = model.cpu()
                        inputs = inputs * 255./255 
                        inputs = inputs.type(torch.uint8).type(torch.float).cpu()           

                    if (save):
                        model = model.cpu()
                        inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                        outputs = F.softmax(model(inputs, save), dim=1)
                        outputs = outputs.to(self.device)
                        npy_to_txt(-1, inputs.int().cpu().detach().numpy())
                    else:
                        outputs = F.softmax(model(inputs), dim=1)
                        outputs = outputs.to(self.device)

                    _, predicted = torch.max(outputs, 1)
                    total += noises.size(0)
                    correct += (predicted == noises).sum().item()
                    # calculate F1 score

                    if minibatch % 20 == 0: 
                        print('[%3d / %3d] accuracy: %.3f' % (minibatch + 1, num_iter,  100 * correct / total))
                        
                        running_loss = 0.0


        if mode == 'validation':
            print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        elif mode == 'testing':
            pass
        if self.args.wandb and mode == 'validation':
            wandb.log({'val_accuracy': 100 * correct / total})
        elif self.args.wandb and mode == 'testing':
            wandb.log({'test_accuracy': 100 * correct / total})

        return(100 * correct / total)


    def validate_test(self, model = None, mode='validation', batch_size = 0, task_id = None,
                 statistics = True, integer = False, save = False, task = None):
        # Validate model

        training_parameters = self.training_parameters
        if (batch_size != 0):
            training_parameters['batch_size'] = batch_size  
        


        data = dataset.AudioGenerator(mode, self.audio_processor, training_parameters, task_id, task = task)
        model.eval()  

        correct = 0
        total = 0

        with torch.no_grad():


            inputs, labels, noises = data[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device) # torch.Size([128, 1, 49, 10])
            # labels = torch.Tensor(labels).long().to(self.device)
            inputs.unsqueeze_(1)
            model = model.to(self.device)  
            noise_types = list(set(noises))
            input_size = inputs.size(0)
            print('input_size', input_size)
            print(noise_types)
            # print('noises: ', noises)
            # correct = {}

            outputs = F.softmax(model(inputs), dim=1)
            outputs = outputs.to(self.device)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # mask_correct = (predicted == labels) # True Positive size: torch.Size([128])
            # mask_task_0 = noises in {0, 1, 2, 3, 4, 5, 6, 7, 8}
            # mask_task_1 = noises in {9, 10, 11}
            # mask_task_2 = noises in {12, 13, 14}
            # mask_task_3 = noises in {15, 16, 17}
            # # intersection of mask_correct and mask_task_0
            # mask_correct_task_0 = mask_correct & mask_task_0
            # mask_correct_task_1 = mask_correct & mask_task_1
            # mask_correct_task_2 = mask_correct & mask_task_2
            # mask_correct_task_3 = mask_correct & mask_task_3
            # print('accuracy task 0', mask_correct_task_0.sum().item())

            if statistics == True:
                # conf_matrix(labels, predicted, self.training_parameters)
                per_noise_accuracy(labels, predicted, noises)


        if mode == 'validation':
            print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        elif mode == 'testing':
            pass
        if self.args.wandb and mode == 'validation':
            wandb.log({'val_accuracy': 100 * correct / total})
        elif self.args.wandb and mode == 'testing':
            wandb.log({'test_accuracy': 100 * correct / total})

        return(100 * correct / total)


    def feat_vis_noise_test(self, model = None, task_id = None,
                 statistics = False):
        # Validate model
        noise_types = ['TBUS_48k', 'OMEETING_48k', 'OHALLWAY_48k', 'NFIELD_48k', 'NPARK_48k', 'NRIVER_48k', \
                       'OOFFICE_48k', 'SPSQUARE_48k', 'DLIVING_48k', 'TCAR_48k', 'STRAFFIC_48k', 'PSTATION_48k', \
                       'DKITCHEN_48k', 'TMETRO_48k', 'DWASHING_48k', 'PCAFETER_48k', 'PRESTO_48k', 'SCAFE_48k' ]
        
        # TBUS_48k: 0, OMEETING_48k: 1, OHALLWAY_48k: 2, NFIELD_48k: 3, NPARK_48k: 4, NRIVER_48k: 5, ...
        noise_dict = {noise_types[i]: i for i in range(len(noise_types))}

        training_parameters = self.training_parameters

        data = dataset_copy.AudioGenerator('testing', self.audio_processor, training_parameters, task_id, task = None)
        model.eval()  

        correct = 0
        total = 0

        # target_words='yes,no,up,down,left,right,on,off,stop,go,'s
                       #2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11

        with torch.no_grad():
            
            all_features = None
        
            inputs, labels, noises = data[0]
            inputs = inputs.to(self.device)
            # print('noises', noises)
            # labels = labels.to(self.device)
            noises = [noise_dict[noise] for noise in noises]
            noises = torch.Tensor(noises).to(self.device).long()
            # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
            # labels = torch.Tensor(labels).long().to(self.device)
            inputs.unsqueeze_(1)
            input_size = inputs.size(0)
            model = model.to(self.device)  
            # print(labels[labels == 2])
            
            
            # outputs = F.softmax(model(inputs), dim=1)
            outputs = model(inputs)
            outputs = outputs.to(self.device)

            # confusion matrix



            features = outputs.detach().cpu().numpy()


            tsne = TSNE(n_components=2)
            embedded_features = tsne.fit_transform(features)
            plt.figure(figsize=(17, 12))

            # colors = [
            #     "#FF0000", "#FF4500", "#32CD32", "#0000FF", "#8A2BE2", "#FF1493", "#00CED1",
            #     "#8B4513", "#000000", "#696969", "#FFD700", "#C0C0C0", "#000080", "#228B22",
            #     "#4B0082", "#DC143C", "#8B0000", "#FF6347"
            # ]
            colors = [
                "#FF0000", "#FF8C00", "#FFD700", "#32CD32", "#0000FF", "#8A2BE2", "#FF1493",
                "#00CED1", "#8B4513", "#000000", "#696969", "#FF69B4", "#C0C0C0", "#000080", 
                "#228B22", "#4B0082", "#DC143C", "#40E0D0"
            ]
            # mapping of the labels to the keyword:
            # target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
    
            #labels = 2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11, ....
            
            
            n_classes = 18

            for i in range(n_classes):
                idx = noises.cpu().numpy() == i
                # plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], color=colors[i], label=f'Class {i}')
                plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], color=colors[i], label=noise_types[i].split('_')[0])


            plt.title('t-SNE Visualization of Model Outputs')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            # plt.legend()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=9)
            mode = self.args.mode if self.args.mode is not None else ""
            method = self.args.method if self.args.method is not None else ""
            remark = self.args.remark if self.args.remark is not None else ""

            timestr = time.strftime("%Y%m%d-%H%M%S")
            run_name = mode + '_' + method + '_' + remark
            path = './images/' + 't-SNE_' + run_name + '_' +  timestr + '.png'
            # plt.savefig(path)


        if self.args.wandb:
            wandb.log({'test_accuracy': 100 * correct / total})

        # return(100 * correct / total)
    
    def feat_vis(self, model = None, task_id = None,
                 statistics = False):
        # Validate model

        training_parameters = self.training_parameters

        data = dataset.AudioGenerator('testing', self.audio_processor, training_parameters, task_id, task = None)
        model.eval()  

        correct = 0
        total = 0

        # target_words='yes,no,up,down,left,right,on,off,stop,go,'s
                       #2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11

        with torch.no_grad():
            
            all_features = None
        
            inputs, labels, _ = data[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
            # labels = torch.Tensor(labels).long().to(self.device)
            inputs.unsqueeze_(1)
            input_size = inputs.size(0)
            model = model.to(self.device)  
            # print(labels[labels == 2])
            
            
            # outputs = F.softmax(model(inputs), dim=1)
            outputs = model(inputs)
            outputs = outputs.to(self.device)

            # only take output that has label greater than 1

            outputs = outputs[labels > 1]
            labels = labels[labels > 1]
            # print(labels[labels == 1])

            features = outputs.detach().cpu().numpy()


            tsne = TSNE(n_components=2)
            embedded_features = tsne.fit_transform(features)
            plt.figure(figsize=(17, 12))

            colors = [
                '#FFFF00', '#FF0000', '#008000', '#808000', '#000080',
                '#0000FF', '#4682B4', '#800080', '#8A2BE2', '#FF00FF',  
                '#000080', '#008080', '#FF4500', '#00CED1', '#228B22', 
                '#191970', '#FFD700', '#6B8E23', '#8B008B', '#9932CC',  
                '#F08080', '#FA8072', '#E9967A', '#FF6347', '#DB7093',  
                '#CD5C5C', '#FFA07A', '#20B2AA', '#7FFFD4', '#00FF7F', 
                '#3CB371', '#00FFFF', '#FF1493', '#7FFF00', '#32CD32',  
                '#FFD700', '#800000', 
            ]

            # mapping of the labels to the keyword:
            # target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
    
            #labels = 2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11, ....
            label_word = ['yes', 'no', 'up', 'down', 'left', \
                        'right', 'on', 'off', 'stop', 'go', \
                        'backward', 'bed', 'bird', 'cat', 'dog', \
                        'eight', 'five', 'follow', 'forward', 'four', \
                        'happy', 'house', 'learn', 'marvin', 'nine', \
                        'one', 'seven', 'sheila', 'six', 'three', \
                        'tree', 'two', 'visual', 'wow', 'zero']
            
            n_classes = 35

            for i in range(2, n_classes+2):
                idx = labels.cpu().numpy() == i
                # plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], color=colors[i], label=f'Class {i}')
                plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], color=colors[i], label=label_word[i-2])


            plt.title('t-SNE Visualization of Model Outputs')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            # plt.legend()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=13)
            mode = self.args.mode if self.args.mode is not None else ""
            method = self.args.method if self.args.method is not None else ""
            remark = self.args.remark if self.args.remark is not None else ""

            run_name = mode + '_' + method + '_' + remark
            path = './images/' + 't-SNE_' + run_name + '.png'
            plt.savefig(path)


        if self.args.wandb:
            wandb.log({'test_accuracy': 100 * correct / total})

        # return(100 * correct / total)
    

    def feat_vis_noise(self, model = None, model_name = None, task_id = None,
                 statistics = False):
        # Validate model

        training_parameters = self.training_parameters

        data = dataset.AudioGenerator('testing', self.audio_processor, training_parameters, task_id, task = None)
        model.eval()  

        correct = 0
        total = 0

        # target_words='yes,no,up,down,left,right,on,off,stop,go,'s
                       #2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11

        with torch.no_grad():
            all_features = None

            inputs, labels, noises = data[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # inputs = torch.Tensor(inputs[:, None, :, :]).to(self.device)
            # labels = torch.Tensor(labels).long().to(self.device)
            inputs.unsqueeze_(1)
            input_size = inputs.size(0)

            # create a noise type set that has 'PCAFETER', 'PRESTO', 'SCAFE'
            noise_task_0 = ['TBUS_48k', 'OMEETING_48k', 'OHALLWAY_48k', 'NFIELD_48k', 'NPARK_48k', 'NRIVER_48k', 'OOFFICE_48k', 'SPSQUARE_48k', 'DLIVING_48k']
            noise_task_1 = ['TCAR_48k', 'STRAFFIC_48k', 'PSTATION_48k']
            noise_task_2 = ['DKITCHEN_48k', 'TMETRO_48k', 'DWASHING_48k']
            noise_task_3 = ['PCAFETER_48k', 'PRESTO_48k', 'SCAFE_48k']
            task_3_set = set(noise_task_3)
            noise_set = set(noise_task_0 + noise_task_1 + noise_task_2 + noise_task_3)
            
            model = model.to(self.device)

            outputs = model(inputs)
            outputs = outputs.to(self.device)

            # outputs = outputs[labels > 1]
            # noises = noises[labels.cpu() > 1]
            # labels = labels[labels > 1]

            # filter out the noise type that is in task 3
            noises = np.array(noises)
            # convert output and labels to numpy array
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            mask = np.isin(noises, list(task_3_set))  # Create a boolean mask
            # outputs = outputs[mask]
            # labels = labels[mask]
            # noises = noises[mask]

            

            colors = [
                '#FFFF00', '#FF0000', '#008000', '#808000', '#000080',
                '#0000FF', '#4682B4', '#800080', '#8A2BE2', '#FF00FF',  
                '#000080', '#008080', '#FF4500', '#00CED1', '#228B22', 
                '#191970', '#FFD700', '#6B8E23', '#8B008B', '#9932CC',  
                '#F08080', '#FA8072', '#E9967A', '#FF6347', '#DB7093',  
                '#CD5C5C', '#FFA07A', '#20B2AA', '#7FFFD4', '#00FF7F', 
                '#3CB371', '#00FFFF', '#FF1493', '#7FFF00', '#32CD32',  
                '#FFD700', '#800000', 
            ]

            # List of noise names
            noises_names = [
                'SCAFE_48k', 'TMETRO_48k', 'NFIELD_48k', 'DWASHING_48k', 'NPARK_48k', 'PRESTO_48k',
                'PSTATION_48k', 'PCAFETER_48k', 'OHALLWAY_48k', 'TBUS_48k', 'DLIVING_48k', 'NRIVER_48k',
                'OOFFICE_48k', 'STRAFFIC_48k', 'TCAR_48k', 'DKITCHEN_48k', 'OMEETING_48k', 'SPSQUARE_48k'
            ]

            label_word = ['yes', 'no', 'up', 'down', 'left', \
                        'right', 'on', 'off', 'stop', 'go', \
                        'backward', 'bed', 'bird', 'cat', 'dog', \
                        'eight', 'five', 'follow', 'forward', 'four', \
                        'happy', 'house', 'learn', 'marvin', 'nine', \
                        'one', 'seven', 'sheila', 'six', 'three', \
                        'tree', 'two', 'visual', 'wow', 'zero']

            tsne = TSNE(n_components=2)
            embedded_features = tsne.fit_transform(outputs)
            plt.figure(figsize=(17, 12))
            word = 'stop'
            word_idx = label_word.index(word) + 2

            for j, noise_name in enumerate(noises_names):
                # idx = (noises == noise_name)
                # idx_class_3 = (labels == 5)
                # mask_intersection = idx & idx_class_3
                # if idx.shape[0] > 0:
                #     plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], color=colors[j], label=noise_name.split('_')[0])
                # Plot points in mask_intersection with the specified color

                idx = (noises == noise_name)
                idx_class_3 = (labels == word_idx)
                idx_noise_task_3 = np.isin(noises, list(task_3_set))
                mask_intersection = idx & idx_class_3

                plt.scatter(
                    embedded_features[mask_intersection, 0], 
                    embedded_features[mask_intersection, 1], 
                    color=colors[j], 
                    label=noise_name.split('_')[0]
                )

                # Plot points not in mask_intersection with light gray color
                not_intersection = idx & ~idx_class_3
                plt.scatter(
                    embedded_features[not_intersection, 0], 
                    embedded_features[not_intersection, 1], 
                    color='lightgray', 
                    # label=noise_name.split('_')[0]
                )



            plt.title('t-SNE Visualization of Model Outputs. Keyword: ' + word)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            # plt.legend()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=9)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            # divide model_name by '_'
            # model_name = 'dil_base_pretrain_20240515-002545_noise_fixed_seed42_snr-8.pth'
            method = model_name.split('_')[1] # 
            path = './images/' + 't-SNE_noise_' + method + '_' + word + '_' + timestr + '.png'
            plt.savefig(path)



            

        if self.args.wandb:
            wandb.log({'test_accuracy': 100 * correct / total})

        # return(100 * correct / total)
    


    
    def noise_data_test(self, model, task_id = None): # Train from scratch
        
        noise_types = ['TBUS_48k', 'OMEETING_48k', 'OHALLWAY_48k', 'NFIELD_48k', 'NPARK_48k', 'NRIVER_48k', \
                       'OOFFICE_48k', 'SPSQUARE_48k', 'DLIVING_48k', 'TCAR_48k', 'STRAFFIC_48k', 'PSTATION_48k', \
                       'DKITCHEN_48k', 'TMETRO_48k', 'DWASHING_48k', 'PCAFETER_48k', 'PRESTO_48k', 'SCAFE_48k' ]
        
        # TBUS_48k: 0, OMEETING_48k: 1, OHALLWAY_48k: 2, NFIELD_48k: 3, NPARK_48k: 4, NRIVER_48k: 5, ...
        noise_dict = {noise_types[i]: i for i in range(len(noise_types))}
        # print(noise_dict)
        
        best_acc = 0
        best_state_dict = None
        num_epochs = 1 if self.args.debug else self.training_parameters['epochs']
        for epoch in range(0, num_epochs):

            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

            data = dataset_copy.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
            print("Data length: " + str(len(data)))
            # take only 10 minibatches
            # data = data[:10]
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            num_iter = 20 if self.training_parameters['debug'] else int(len(data))
            num_print = 50

            for minibatch in range(num_iter): 
            
                inputs, labels, noises = data[0]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # convert noise to integer according to noise_dict and convert to tensor
                noises = [noise_dict[noise] for noise in noises]
                noises = torch.Tensor(noises).to(self.device).long()

                inputs.unsqueeze_(1)

                # noise_types = list(set(noises))
                # print(noise_types)
                if inputs.size(0) == 0:
                    print('number of minibatch: ', minibatch)
                    break
                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)

                loss = self.criterion(outputs, noises)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += noises.size(0)
                correct += (predicted == noises).sum().item()

                if self.args.wandb:
                    wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                # Print information every 50 minibatches
                if minibatch % num_print == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                    
                    running_loss = 0.0

            tmp_acc = self.validate_noise(model, 'validation', task_id = task_id)
            if self.args.early_stop:
                # Save best performing network
                if (tmp_acc > best_acc):
                    best_acc = tmp_acc
                    best_ep  = epoch
                    best_state_dict = model.state_dict()

                patience = int(self.config['patience'])
                if (epoch >= best_ep + patience):
                    break
            else:
                best_state_dict = model.state_dict()
        

        timestr = time.strftime("%Y%m%d-%H%M%S")

        model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '_' + self.args.remark + '.pth'
        PATH = './models/' + model_name

        torch.save(best_state_dict, PATH)
        print('model saved at: ', PATH)


        
    def noise_data_diversity(self, model, task_id = None): # Train from scratch
        
        noise_types = ['TBUS_48k', 'OMEETING_48k', 'OHALLWAY_48k', 'NFIELD_48k', 'NPARK_48k', 'NRIVER_48k', \
                       'OOFFICE_48k', 'SPSQUARE_48k', 'DLIVING_48k', 'TCAR_48k', 'STRAFFIC_48k', 'PSTATION_48k', \
                       'DKITCHEN_48k', 'TMETRO_48k', 'DWASHING_48k', 'PCAFETER_48k', 'PRESTO_48k', 'SCAFE_48k' ]
        
        # TBUS_48k: 0, OMEETING_48k: 1, OHALLWAY_48k: 2, NFIELD_48k: 3, NPARK_48k: 4, NRIVER_48k: 5, ...
        noise_dict = {noise_types[i]: i for i in range(len(noise_types))}
        # print(noise_dict)
        
        best_acc = 0
        best_state_dict = None
        num_epochs = 1 if self.args.debug else self.training_parameters['epochs']
        data = dataset_copy.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
        print("Data length: " + str(len(data)))
        num_iter = 20 if self.training_parameters['debug'] else len(data)
        for minibatch in range(num_iter): 
            
            inputs, labels, noises = data[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # convert noise to integer according to noise_dict and convert to tensor
            noises = [noise_dict[noise] for noise in noises]
            noises = torch.Tensor(noises).to(self.device).long()

            inputs.unsqueeze_(1)
            print('inputs size: ', inputs.size())
        

    def train(self, model, task_id = None): # Train from scratch
    
        
        best_acc = 0
        best_state_dict = None
        num_epochs = 1 if self.args.debug else self.training_parameters['epochs']
        for epoch in range(0, num_epochs):

            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
            print("Data length: " + str(len(data)))
            # take only 10 minibatches
            # data = data[:10]
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            num_iter = 20 if self.training_parameters['debug'] else len(data)
            num_print = 50

            for minibatch in range(num_iter): 
            
                inputs, labels, noises = data[0]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                # labels = torch.Tensor(labels).to(self.device).long()
                inputs.unsqueeze_(1)

                noise_types = list(set(noises))
                # print(noise_types)
                if inputs.size(0) == 0:
                    print('number of minibatch: ', minibatch)
                    break
                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if self.args.wandb:
                    wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                # Print information every 50 minibatches
                if minibatch % num_print == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                    
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation', task_id = task_id)
            if self.args.early_stop:
                # Save best performing network
                if (tmp_acc > best_acc):
                    best_acc = tmp_acc
                    best_ep  = epoch
                    best_state_dict = model.state_dict()

                patience = int(self.config['patience'])
                if (epoch >= best_ep + patience):
                    break
            else:
                best_state_dict = model.state_dict()
        

        timestr = time.strftime("%Y%m%d-%H%M%S")

        model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '_' + self.args.remark + '.pth'
        PATH = './models/' + model_name

        torch.save(best_state_dict, PATH)
        print('model saved at: ', PATH)


    def finetune(self, model, task_id = None): 
    
        
        best_acc = 0
        best_state_dict = None
        num_epochs = 1 if self.args.debug else self.config['epochs_CL']
        for epoch in range(0, num_epochs):

            print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
            print("Data length: " + str(len(data)))
            # take only 10 minibatches
            # data = data[:10]
            for param in model.parameters():
                param.requires_grad = False
            model.fc1.weight.requires_grad = True
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            num_iter = 20 if self.training_parameters['debug'] else len(data)

            for minibatch in range(num_iter): 
            
                inputs, labels, noises = data[0]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                # labels = torch.Tensor(labels).to(self.device).long()
                inputs.unsqueeze_(1)
                if inputs.size(0) == 0:
                    print('number of minibatch: ', minibatch)
                    break
                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print information every 50 minibatches
                if minibatch % 50 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / 10, 100 * correct / total))
                    if self.args.wandb:
                        wandb.log({'loss': running_loss / 10, 'accuracy': 100 * correct / total})   
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation', task_id = task_id)

        return model
    

    def ER(self, 
                model, 
                memory_buffer, 
                task_id = None,):
        self.loss_kd = None
        # Train model
        best_acc = 0
        num_epochs = 1 if self.args.debug else self.config['epochs_CL']
        # memory_buffer.to_device(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(self.device)
        model_copy.eval()

        for epoch in range(0, num_epochs):

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
            print("Data length: " + str(len(data))) # 288
            
            print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            num_iter = len(data)

            num_print = 20

            for minibatch in range(num_iter): 

                inputs, labels, noise = data[0]
                inputs = inputs.to(self.device) # torch.Size([128, 1, 49, 10])
                labels = labels.to(self.device) # torch.Size([128])
                
                # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device) # torch.Size([128, 1, 49, 10])
                # labels = torch.Tensor(labels).to(self.device).long()
                inputs.unsqueeze_(1)
                if self.args.mixup:
                    # Apply Mixup data augmentation
                    inputs, labels_a, labels_b, lam = self.mixup_data(inputs, labels)
                    
                    # Convert to autograd.Variable if not already
                    inputs, labels_a, labels_b = map(torch.autograd.Variable, (inputs, labels_a, labels_b))

                if inputs.size(0) == 0: # inputsize torch.Size([0, 1, 49, 10])
                    break
                input_size = int(inputs.size(0))
                # update memory buffer in the last epoch (to ensure that the memory buffer
                # is not updated multiple times)
                if epoch == num_epochs - 1:
                    if self.args.record_noise_type:
                        # print('noise type: ', noise)
 
                        memory_buffer.add_data(inputs, labels, noise_type = noise)
                        # noise_type = memory_buffer.get_noise_type()
                        # # noise_type is a dict. calculate the number of values in the dict
                        # sum_values = sum([len(value) for value in noise_type.values()])
                        # print('number of noise types: ', sum_values)

                        # # print('noise type: ', noise_type)
                    else:
                        memory_buffer.add_data(inputs, labels)

                
                samples_inputs, samples_labels = memory_buffer.get_data(input_size)
                samples_labels = samples_labels.long()
                # move to device
                samples_inputs = samples_inputs.to(self.device)
                samples_labels = samples_labels.to(self.device)
                # if self.args.augmentation:
                #     samples_inputs = augment_mfcc(samples_inputs).to(self.device)
                inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])
                if self.args.augmentation:
                    inputs = augment_mfcc(inputs).to(self.device)

                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)

                if self.args.kd: # knowledge distillation
                    if not (epoch == num_epochs - 1):
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])
                        # loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                    else:
                        
                        logits = model(inputs)
                        logits_new = logits[input_size:].clone().detach()
                        outputs = F.softmax(logits, dim=1)
                        with torch.no_grad():
                            logits_old = model_copy(inputs.clone().detach())
                            logits_old = logits_old[input_size:].clone().detach()
                            logits_new = logits_new.to(self.device)
                            logits_old = logits_old.to(self.device)
                            self.loss_kd = self.criterion_kd(logits_new, logits_old, temperature=2.0)
                        if self.loss_kd is None:
                            loss = self.criterion(outputs, labels)
                        else:
                            loss = self.criterion(outputs, labels) + float(self.config['kd_alpha']) * self.loss_kd
                            if minibatch % 50 == 0: 
                                print('kd loss: ', self.loss_kd, 'loss:', loss)
                        # loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                        # model.eval()
                        # with torch.no_grad():
                        #     # logits_samples_new = model(inputs)[input_size:]
                        #     # # calculate knowledge distillation loss between logits_samples and logits_samples_new
                        #     # self.loss_kd = self.criterion_kd(logits_samples, logits_samples_new, temperature=2.0)
                        #     # # print('kd loss: ', self.loss_kd, 'loss:', loss)
                        #     self.loss_kd = self.criterion_kd(logits_samples, logits_old, temperature=2.0)
                        # model.train()
                else:
                    outputs = F.softmax(model(inputs), dim=1)
                    if not self.args.mixup:
                        loss = self.criterion(outputs, labels) # torch.Size([256])
                    else:
                        loss = self.mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    # loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()


                

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if self.args.wandb:
                    wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                # Print information every 50 minibatches
                if minibatch % num_print == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                    
                    running_loss = 0.0
            

            tmp_acc = self.validate(model, 'validation', task_id = task_id)


        return model, memory_buffer


    def ER_custom(self, 
                model, 
                memory_buffer, 
                task_id = None,):
    
        num_epochs = 1 if self.args.debug else self.config['epochs_CL']
        # memory_buffer.to_device(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        buffer_data_ratio = 1
        for epoch in range(0, num_epochs):

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
            print("Data length: " + str(len(data))) # 288
            
            print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            # num_iter = 20 if self.training_parameters['debug'] else len(data)
            num_iter = len(data)
            num_print = 20

            for minibatch in range(num_iter): 

                inputs, labels, _ = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device) # torch.Size([128, 1, 49, 10])
                labels = torch.Tensor(labels).to(self.device).long()
                if inputs.size(0) == 0: # inputsize torch.Size([0, 1, 49, 10])
                    break
                input_size = int(inputs.size(0)) # 128
                # update memory buffer in the last epoch (to ensure that the memory buffer
                # is not updated multiple times)
                
                # check input shape
                # sample_size = int(input_size*buffer_data_ratio)
                # if sample_size > 1:
                #     samples_inputs, samples_labels = memory_buffer.get_data(sample_size)
                # else:
                #     samples_inputs, samples_labels = memory_buffer.get_data(input_size)

                # augmentation for samples: Clipping Distortion, Time Mask, Shift, and Random Frequency Mask on MFCC
                # if self.args.augmentation:
                #     augmentor = RandomAugmentor(device=self.device)
                #     samples_inputs = augmentor.apply(samples_inputs)

                samples_inputs, samples_labels = memory_buffer.get_data(input_size)

                samples_labels = samples_labels.long()
                # samples_labels = torch.Tensor(samples_labels).to(self.device).long()

                inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])
                # print('epoch', epoch, 'minibatch', minibatch)
                # Zero out the parameter gradients after each mini-batch
                
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)
                loss = self.criterion(outputs, labels) # torch.Size([256])

 
                loss_per_class = {}
                # group loss[input_size:] by labels
                for i in range(input_size, len(loss)):
                    if labels[i].item() in loss_per_class:
                        loss_per_class[labels[i].item()].append(loss[i].item())
                    else:
                        loss_per_class[labels[i].item()] = [loss[i].item()]
                # print('loss_per_class', loss_per_class)
                # get the mean loss for each class
                for key in loss_per_class:
                    loss_per_class[key] = sum(loss_per_class[key])/len(loss_per_class[key])

                # sort the class by loss, ascending order
                loss_per_class = dict(sorted(loss_per_class.items(), key=lambda item: item[1]))
                
                # only get the keys.
                # max_loss_class = list(loss_per_class.keys())
                print('max_loss_class', loss_per_class)
                # if epoch == num_epochs - 1:
                if epoch == 0:
                    memory_buffer.add_data(inputs[:input_size], labels[:input_size], loss_per_class)

                # print('max_loss_class', max_loss_class)
                # mean loss
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                TP_all = (predicted == labels) # True Positive size: torch.Size([256])
                TP_inputs = TP_all[:input_size] # torch.Size([128])
                TP_samples = TP_all[input_size:] # torch.Size([128])
                # Calculate accuracy for inputs
                accuracy_inputs = TP_inputs.float().mean().item() # 

                # Calculate accuracy for samples
                accuracy_samples = TP_samples.float().mean().item()
                # Calculate accuracy ratio
                if accuracy_samples == 0:
                    accuracy_ratio = 2
                else:
                    accuracy_ratio = accuracy_inputs / accuracy_samples 

                # if accuracy_ratio is high, more data from memory buffer should be used.
                # if accuracy_ratio is low, less data from memory buffer should be used.

                buffer_data_ratio = accuracy_ratio
                # print('accuracy_ratio: ', accuracy_ratio)
                correct += (predicted == labels).sum().item()
                
                if self.args.wandb:
                    wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                # Print information every 50 minibatches
                if minibatch % num_print == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                    
                    running_loss = 0.0
                # Print information every 50 minibatches
                # if minibatch % 50 == 0: 
                #     print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / 10, 100 * correct / total))
                #     if self.args.wandb:
                #         wandb.log({'loss': running_loss / 10, 'accuracy': 100 * correct / total})   
                #     running_loss = 0.0
                

            tmp_acc = self.validate(model, 'validation', task_id = task_id)

        
        # model_name = self.args.method + '_' + self.args.task + '_model.pth'
        # PATH = './models/' + model_name
        # torch.save(model.state_dict(), PATH)


        return model, memory_buffer


    def ER_LAECB(self, 
                model, 
                memory_buffer, 
                task_id = None,):
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        # memory_buffer = memory_buffer.to(self.device)

        # Train model
        best_acc = 0

        if task_id == 'cil_task_0' or task_id == 'dil_task_0':

            if self.args.load_model:
                print('Training from pre-trained model')
                print('task_id', task_id)
                num_epochs = 1
                best_ep = -1
                flag_to_stop = False
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288

                    print("Epoch: " + str(epoch+1) +"/1")
                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    
                    total_time = 0
                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, noise = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            if self.args.record_noise_type:
                                memory_buffer.add_data(inputs, temp_loss, labels, noise_type = noise)
                                # noise_type = memory_buffer.get_noise_type()
                                # sum_values = sum([len(value) for value in noise_type.values()])
                                # print('number of noise types: ', sum_values)
                                # concatenate the noise type
                                # print('noise type: ', noise_type)
                            else:
                                memory_buffer.add_data(inputs, temp_loss, labels)
                            

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:
                                model.eval()
                                with torch.no_grad():
                                    samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                    samples_inputs = samples_inputs.to(self.device)
                                    samples_labels = samples_labels.to(self.device).long()

                                    # Use the model copy to compute logits
                                    outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                    # Compute loss using the logits from the model copy
                                    loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                                    memory_buffer.update_loss(loss_samples, samples_labels)
                                    # print('Loss updated at iteration:', minibatch)
                                model.train()


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer
            
            else :
                print('Training from scratch')
                print('task_id', task_id)
                num_epochs = 1 if self.args.debug else self.config['epochs']
                best_ep = -1
                flag_to_stop = False

                total_time = 0
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288
                    
                    if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                    else:
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    

                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            start_time = time.time()
                            memory_buffer.add_data(inputs, temp_loss, labels)
                            end_time = time.time()
                            total_time = total_time + (end_time - start_time)
                            avg_time = total_time / (minibatch + 1)
                            
                            if self.args.wandb:
                                wandb.log({'AVG Time for adding data': avg_time})

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:
                                model.eval()
                                with torch.no_grad():
                                    samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                    samples_inputs = samples_inputs.to(self.device)
                                    samples_labels = samples_labels.to(self.device).long()

                                    # Use the model copy to compute logits
                                    outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                    # Compute loss using the logits from the model copy
                                    loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                                    memory_buffer.update_loss(loss_samples, samples_labels)
                                    # print('Loss updated at iteration:', minibatch)
                                model.train()


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer

        else:
            num_epochs = 1 if self.args.debug else self.config['epochs_CL']
   
            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                else:
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                model.train()
                self.scheduler.step()

                running_loss = 0.0
                total = 0
                correct = 0   

                # num_iter = 20 if self.training_parameters['debug'] else len(data)
                num_iter = len(data)
                num_print = 20

                for minibatch in range(num_iter): 

                    inputs, labels, noise = data[0]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).to(self.device).long()
                    inputs.unsqueeze_(1)
                    if inputs.size(0) == 0:
                        break
                    input_size = int(inputs.size(0))
                    # update memory buffer in the last epoch (to ensure that the memory buffer
                    # is not updated multiple times)
                    
                    samples_inputs, samples_labels = memory_buffer.get_data(input_size) 
    
                    # samples_inputs = samples_inputs.to(self.device)
                    # samples_labels = samples_labels.to(self.device).long()
                    samples_labels = samples_labels.long()
                    # samples_labels = torch.Tensor(samples_labels).to(self.device).long()
                    inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                    labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])

                    if self.args.augmentation:
                        inputs = augment_mfcc(inputs).to(self.device)
                    
                    # Zero out the parameter gradients after each mini-batch
                    self.optimizer.zero_grad()

                    # Train, compute loss, update optimizer
                    model = model.to(self.device)
                    outputs = F.softmax(model(inputs), dim=1)
                    loss = self.criterion(outputs, labels) # torch.Size([256])

                    loss_samples = loss[input_size:].clone().detach()
                    loss_new = loss[:input_size].clone().detach()

                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()
                    
                    memory_buffer.update_loss(loss_samples, samples_labels)
                    if epoch == num_epochs - 1:
                    # if epoch == 0:
                        # add only first 128 input, loss and labels
                        if self.args.record_noise_type:
   
                            memory_buffer.add_data(inputs[:input_size], loss_new, labels[:input_size], noise_type = noise[:input_size])
                            # if minibatch % 50 == 0:

                            #     noise_type = memory_buffer.get_noise_type()
                            #     # concatenate the noise type
                            #     sum_values = sum([len(value) for value in noise_type.values()])
                            #     print('number of noise types: ', sum_values)
                        else:
                            memory_buffer.add_data(inputs[:input_size], loss_new, labels[:input_size])

                        model.eval()
                        with torch.no_grad():
                            # samples_inputs, samples_labels = memory_buffer.get_new_data()
                            samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                            samples_inputs = samples_inputs.to(self.device)
                            samples_labels = samples_labels.to(self.device).long()
                            if self.args.augmentation:
                                samples_inputs = augment_mfcc(samples_inputs).to(self.device)
                            # Use the model copy to compute logits
                            outputs_samples = F.softmax(model(samples_inputs), dim=1)

                            # Compute loss using the logits from the model copy
                            loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                            memory_buffer.update_loss(loss_samples, samples_labels)
                            # print('Loss updated at iteration:', minibatch)
                        model.train()
                        
                    # mean loss
                    # loss = loss.mean()
                    
                    # loss.item() # shape: torch.Size([256])

                    # Compute training statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.wandb:
                        wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                    # Print information every 50 minibatches
                    if minibatch % num_print == 0: 
                        print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                        
                        running_loss = 0.0
                    


                tmp_acc = self.validate(model, 'validation', task_id = task_id)
    

            return model, memory_buffer


    def ER_GAECB(self, 
                model, 
                memory_buffer, 
                task_id = None,):
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        # memory_buffer = memory_buffer.to(self.device)

        # Train model
        best_acc = 0

        if task_id == 'cil_task_0' or task_id == 'dil_task_0':

            if self.args.load_model:
                print('Training from pre-trained model')
                print('task_id', task_id)
                num_epochs = 1
                best_ep = -1
                flag_to_stop = False
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288

                    print("Epoch: " + str(epoch+1) +"/1")
                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    
                    total_time = 0
                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        # print('inputs', inputs.size())
                        # print('labels', labels.size())
                        if inputs.size(0) == 0:
                            break

                        if minibatch == 10 and self.args.debug:
                            break
                        
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()

                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            
                            memory_buffer.add_data(inputs, temp_loss, labels)

                            for param in model.parameters():
                                param.requires_grad = True
                            samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                            samples_inputs = samples_inputs.to(self.device)
                            samples_labels = samples_labels.to(self.device).long()

                            samples_inputs.requires_grad_(True)

                            outputs_samples = F.softmax(model(samples_inputs), dim=1)

                            # Compute loss using the logits from the model copy
                            loss_samples = self.criterion(outputs_samples, samples_labels) # torch.Size([256])

                            inputs_grad = torch.autograd.grad(loss_samples.mean(), samples_inputs, retain_graph=True, allow_unused=True)[0]
                            inputs_grad_flat = inputs_grad.view(inputs_grad.size(0), -1) 
                            # Compute the L2 norm iteratively over all non-batch dimensions
                            sample_grad_norms = torch.norm(inputs_grad_flat, dim=1)
          
                            memory_buffer.update_loss(sample_grad_norms, samples_labels)



                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer
            
            else :
                print('Training from scratch')
                print('task_id', task_id)
                num_epochs = 1 if self.args.debug else self.config['epochs']
                best_ep = -1
                flag_to_stop = False

                total_time = 0
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288
                    
                    if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                    else:
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    

                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            start_time = time.time()
                            memory_buffer.add_data(inputs, temp_loss, labels)
                            end_time = time.time()
                            total_time = total_time + (end_time - start_time)
                            avg_time = total_time / (minibatch + 1)
                            
                            if self.args.wandb:
                                wandb.log({'AVG Time for adding data': avg_time})

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:

                                samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                samples_inputs = samples_inputs.to(self.device)
                                samples_labels = samples_labels.to(self.device).long()

                                # Use the model copy to compute logits
                                outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                # Compute loss using the logits from the model copy
                                loss_samples = self.criterion(outputs_samples, samples_labels)

                                inputs_grad = torch.autograd.grad(loss_samples.mean(), samples_inputs, retain_graph=True, allow_unused=True)[0]
                                inputs_grad_flat = inputs_grad.view(inputs_grad.size(0), -1) 
                                # Compute the L2 norm iteratively over all non-batch dimensions
                                sample_grad_norms = torch.norm(inputs_grad_flat, dim=1)

                                memory_buffer.update_loss(sample_grad_norms, samples_labels)
                                # print('Loss updated at iteration:', minibatch)
               


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer

        else:
            num_epochs = 2 if self.args.debug else self.config['epochs_CL']
   
            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                else:
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                model.train()
                self.scheduler.step()

                running_loss = 0.0
                total = 0
                correct = 0   

                # num_iter = 20 if self.training_parameters['debug'] else len(data)
                num_iter = len(data)
                num_print = 20

                for minibatch in range(num_iter): 

                    # print('minibatch', minibatch)

                    inputs, labels, _ = data[0]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # inputs: torch.Size([N, 49, 10])
                    # unsqueeze inputs to add channel dimension to torch.Size([N, 1, 49, 10])
                    inputs.unsqueeze_(1)
                    # print('inputs', inputs.size())
                    # print('labels', labels.size())

                   

                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).to(self.device).long()
                    if inputs.size(0) == 0:
                        break
                    input_size = int(inputs.size(0))
                    # update memory buffer in the last epoch (to ensure that the memory buffer
                    # is not updated multiple times)
                    
                    samples_inputs, samples_labels = memory_buffer.get_data(input_size) 

                    # samples_inputs.requires_grad_(True)

                    samples_labels = samples_labels.long()
                    # samples_labels = torch.Tensor(samples_labels).to(self.device).long()
                    inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                    labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])

                    # Zero out the parameter gradients after each mini-batch
                    self.optimizer.zero_grad()

                    # Train, compute loss, update optimizer
                    model = model.to(self.device)
                    for param in model.parameters():
                        param.requires_grad = True

                    outputs = F.softmax(model(inputs), dim=1)
                    loss = self.criterion(outputs, labels) # torch.Size([256])

                    loss_samples = loss[input_size:]
                    loss_new = loss[:input_size]

                    inputs_copy = inputs.clone().requires_grad_(True)

                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()

                    

                    inputs_temp = inputs.clone().requires_grad_(True)
                    # Use the model copy to compute logits
                    outputs_temp = F.softmax(model(inputs_temp), dim=1)
                    labels_temp = labels.clone()

                    # Compute loss using the logits from the model copy
                    loss_samples = self.criterion(outputs_temp, labels_temp).mean()

                    inputs_grad = torch.autograd.grad(loss_samples, inputs_temp, retain_graph=True, allow_unused=True)[0]
                    inputs_grad_flat = inputs_grad.view(inputs_grad.size(0), -1) 
                    # Compute the L2 norm iteratively over all non-batch dimensions
                    grad_norms = torch.norm(inputs_grad_flat, dim=1)
                    # print('sample_grad_norms', sample_grad_norms)
                    sample_grad_norms = grad_norms[input_size:]
                    new_grad_norms = grad_norms[:input_size]

                    memory_buffer.update_loss(sample_grad_norms, samples_labels)
                    if epoch == num_epochs - 1:

                        memory_buffer.add_data(inputs[:input_size], new_grad_norms, labels[:input_size])

         

                    # if epoch == num_epochs - 1:
                    # # if epoch == 0:
                    #     # add only first 128 input, loss and labels
                    #     memory_buffer.add_data(inputs[:input_size], loss_new, labels[:input_size])

                        # # samples_inputs, samples_labels = memory_buffer.get_new_data()
                        # samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                        # samples_inputs = samples_inputs.to(self.device)
                        # samples_labels = samples_labels.to(self.device).long()

                        # samples_inputs.requires_grad_(True)

                        # # Use the model copy to compute logits
                        # outputs_samples = F.softmax(model(samples_inputs), dim=1)

                        # # Compute loss using the logits from the model copy
                        # loss_samples = self.criterion(outputs_samples, samples_labels)

                        # inputs_grad = torch.autograd.grad(loss_samples.mean(), samples_inputs, retain_graph=True, allow_unused=True)[0]
                        # inputs_grad_flat = inputs_grad.view(inputs_grad.size(0), -1) 
                        # # Compute the L2 norm iteratively over all non-batch dimensions
                        # sample_grad_norms = torch.norm(inputs_grad_flat, dim=1)
                        # # print('sample_grad_norms hhhhhh', sample_grad_norms)

                        # memory_buffer.update_loss(sample_grad_norms, samples_labels)



                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.wandb:
                        wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                    # Print information every 50 minibatches
                    if minibatch % num_print == 0: 
                        print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                        
                        running_loss = 0.0
                    


                tmp_acc = self.validate(model, 'validation', task_id = task_id)
    

            return model, memory_buffer


    def ER_LDAECB(self, 
                model, 
                memory_buffer, 
                task_id = None,):
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        # memory_buffer = memory_buffer.to(self.device)

        # Train model
        best_acc = 0

        if task_id == 'cil_task_0' or task_id == 'dil_task_0':

            if self.args.load_model:
                print('Training from pre-trained model')
                print('task_id', task_id)
                num_epochs = 1
                best_ep = -1
                flag_to_stop = False
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288

                    print("Epoch: " + str(epoch+1) +"/1")
                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    
                    total_time = 0
                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        inputs.unsqueeze_(1)
                        # print('inputs', inputs.size())
                        # inputs = augment_mfcc(inputs).to(self.device)
                        # print('inputs', inputs.size())
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            
                            memory_buffer.add_data(inputs, temp_loss, labels)
                            

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:
                                model.eval()
                                with torch.no_grad():
                                    samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                    samples_inputs = samples_inputs.to(self.device)
                                    samples_labels = samples_labels.to(self.device).long()

                                    # Use the model copy to compute logits
                                    outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                    # Compute loss using the logits from the model copy
                                    loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                                    memory_buffer.update_loss(loss_samples, samples_labels)
                                    # print('Loss updated at iteration:', minibatch)
                                model.train()


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer
            
            else :
                print('Training from scratch')
                print('task_id', task_id)
                num_epochs = 1 if self.args.debug else self.config['epochs']
                best_ep = -1
                flag_to_stop = False

                total_time = 0
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288
                    
                    if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                    else:
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    

                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            start_time = time.time()
                            memory_buffer.add_data(inputs, temp_loss, labels)
                            end_time = time.time()
                            total_time = total_time + (end_time - start_time)
                            avg_time = total_time / (minibatch + 1)
                            
                            if self.args.wandb:
                                wandb.log({'AVG Time for adding data': avg_time})

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:
                                model.eval()
                                with torch.no_grad():
                                    samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                    samples_inputs = samples_inputs.to(self.device)
                                    samples_labels = samples_labels.to(self.device).long()

                                    # Use the model copy to compute logits
                                    outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                    # Compute loss using the logits from the model copy
                                    loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                                    memory_buffer.update_loss(loss_samples, samples_labels)
                                    # print('Loss updated at iteration:', minibatch)
                                model.train()


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer

        else:
            num_epochs = 1 if self.args.debug else self.config['epochs_CL']
   
            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                else:
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                model.train()
                self.scheduler.step()

                running_loss = 0.0
                total = 0
                correct = 0   

                # num_iter = 20 if self.training_parameters['debug'] else len(data)
                num_iter = len(data)
                num_print = 20

                for minibatch in range(num_iter): 

                    inputs, labels, _ = data[0]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).to(self.device).long()
                    inputs.unsqueeze_(1)
                    
                    if inputs.size(0) == 0:
                        break
                    input_size = int(inputs.size(0))
                    # update memory buffer in the last epoch (to ensure that the memory buffer
                    # is not updated multiple times)
                    
                    samples_inputs, samples_labels = memory_buffer.get_data(input_size) 
    
                    # samples_inputs = samples_inputs.to(self.device)
                    # samples_labels = samples_labels.to(self.device).long()
                    samples_labels = samples_labels.long()
                    # samples_labels = torch.Tensor(samples_labels).to(self.device).long()
                    inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                    labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])

                    if self.args.augmentation:
                        inputs = augment_mfcc(inputs).to(self.device)
                    
                    # Zero out the parameter gradients after each mini-batch
                    self.optimizer.zero_grad()

                    # Train, compute loss, update optimizer
                    model = model.to(self.device)
                    outputs = F.softmax(model(inputs), dim=1)
                    loss = self.criterion(outputs, labels) # torch.Size([256])

                    loss_samples = loss[input_size:].clone().detach()
                    loss_new = loss[:input_size].clone().detach()

                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()
                    
                    memory_buffer.update_loss(loss_samples, samples_labels)
                    if epoch == num_epochs - 1:
                    # if epoch == 0:
                        # add only first 128 input, loss and labels
                        memory_buffer.add_data(inputs[:input_size], loss_new, labels[:input_size])

                        model.eval()
                        with torch.no_grad():
                            # samples_inputs, samples_labels = memory_buffer.get_new_data()
                            samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                            samples_inputs = samples_inputs.to(self.device)
                            samples_labels = samples_labels.to(self.device).long()
                            # Use the model copy to compute logits
                            outputs_samples = F.softmax(model(samples_inputs), dim=1)

                            # Compute loss using the logits from the model copy
                            loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                            memory_buffer.update_loss(loss_samples, samples_labels)
                            # print('Loss updated at iteration:', minibatch)
                        model.train()
                        
                    # mean loss
                    # loss = loss.mean()
                    
                    # loss.item() # shape: torch.Size([256])

                    # Compute training statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.wandb:
                        wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                    # Print information every 50 minibatches
                    if minibatch % num_print == 0: 
                        print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                        
                        running_loss = 0.0
                    


                tmp_acc = self.validate(model, 'validation', task_id = task_id)
    

            return model, memory_buffer


    def ER_LDAAECB(self, 
                model, 
                memory_buffer, 
                task_id = None,):
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        # memory_buffer = memory_buffer.to(self.device)

        # Train model
        best_acc = 0

        if task_id == 'cil_task_0' or task_id == 'dil_task_0':

            if self.args.load_model:
                print('Training from pre-trained model')
                print('task_id', task_id)
                num_epochs = 1
                best_ep = -1
                flag_to_stop = False
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288

                    print("Epoch: " + str(epoch+1) +"/1")
                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    
                    total_time = 0
                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            # print('inputs', inputs.size(), 'temp_loss', temp_loss.size(), 'labels', labels.size())
                            memory_buffer.add_data(inputs, temp_loss, labels)
                            _, predicted = torch.max(outputs.data, 1)
                            TP_all = (predicted == labels) # True Positive size: torch.Size([256])
                            classes = labels.unique()
                            accuracy_per_class = {}
                            for c in classes:
                                TP = TP_all[labels == c]
                                accuracy_per_class[c.item()] = TP.float().mean().item()
                            memory_buffer.update_acc(accuracy_per_class)
                            

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:
                                model.eval()
                                with torch.no_grad():
                                    samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                    samples_inputs = samples_inputs.to(self.device)
                                    samples_labels = samples_labels.to(self.device).long()

                                    # Use the model copy to compute logits
                                    outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                    # Compute loss using the logits from the model copy
                                    loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                                    memory_buffer.update_loss(loss_samples, samples_labels)
                                    # print('Loss updated at iteration:', minibatch)
                                model.train()


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer
            
            else :
                print('Training from scratch')
                print('task_id', task_id)
                num_epochs = 1 if self.args.debug else self.config['epochs']
                best_ep = -1
                flag_to_stop = False

                total_time = 0
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288
                    
                    if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                    else:
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    

                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            start_time = time.time()
                            memory_buffer.add_data(inputs, temp_loss, labels)
                            end_time = time.time()
                            total_time = total_time + (end_time - start_time)
                            avg_time = total_time / (minibatch + 1)
                            
                            if self.args.wandb:
                                wandb.log({'AVG Time for adding data': avg_time})

                            if minibatch >= self.config['memory_buffer_size'] / self.config['batch_size']:
                                model.eval()
                                with torch.no_grad():
                                    samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                                    samples_inputs = samples_inputs.to(self.device)
                                    samples_labels = samples_labels.to(self.device).long()

                                    # Use the model copy to compute logits
                                    outputs_samples = F.softmax(model(samples_inputs), dim=1)

                                    # Compute loss using the logits from the model copy
                                    loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                                    memory_buffer.update_loss(loss_samples, samples_labels)
                                    # print('Loss updated at iteration:', minibatch)
                                model.train()


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer

        else:
            num_epochs = 1 if self.args.debug else self.config['epochs_CL']
   
            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                else:
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                model.train()
                self.scheduler.step()

                running_loss = 0.0
                total = 0
                correct = 0   

                # num_iter = 20 if self.training_parameters['debug'] else len(data)
                num_iter = len(data)
                num_print = 20

                for minibatch in range(num_iter): 

                    inputs, labels, _ = data[0]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).to(self.device).long()
                    inputs.unsqueeze_(1)
                    if inputs.size(0) == 0:
                        break
                    input_size = int(inputs.size(0))
                    # update memory buffer in the last epoch (to ensure that the memory buffer
                    # is not updated multiple times)
                    
                    samples_inputs, samples_labels = memory_buffer.get_data(input_size) 
    
                    # samples_inputs = samples_inputs.to(self.device)
                    # samples_labels = samples_labels.to(self.device).long()
                    samples_labels = samples_labels.long()
                    # samples_labels = torch.Tensor(samples_labels).to(self.device).long()
                    inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                    labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])
                    
                    # Zero out the parameter gradients after each mini-batch
                    self.optimizer.zero_grad()

                    # Train, compute loss, update optimizer
                    model = model.to(self.device)
                    outputs = F.softmax(model(inputs), dim=1)
                    loss = self.criterion(outputs, labels) # torch.Size([256])

                    loss_samples = loss[input_size:].clone().detach()
                    loss_new = loss[:input_size].clone().detach()

                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()
                    
                    memory_buffer.update_loss(loss_samples, samples_labels)
                    if epoch == num_epochs - 1:
                    # if epoch == 0:
                        # add only first 128 input, loss and labels
                        memory_buffer.add_data(inputs[:input_size], loss_new, labels[:input_size])
                        _, predicted = torch.max(outputs.data, 1)
                        TP_all = (predicted == labels) # True Positive size: torch.Size([256])
                        classes = labels.unique()
                        accuracy_per_class = {}
                        for c in classes:
                            TP = TP_all[labels == c]
                            accuracy_per_class[c.item()] = TP.float().mean().item()
                        memory_buffer.update_acc(accuracy_per_class)

                        model.eval()
                        with torch.no_grad():
                            # samples_inputs, samples_labels = memory_buffer.get_new_data()
                            samples_inputs, samples_labels = memory_buffer.get_data(self.config['memory_buffer_size'])
                            samples_inputs = samples_inputs.to(self.device)
                            samples_labels = samples_labels.to(self.device).long()
                            # Use the model copy to compute logits
                            outputs_samples = F.softmax(model(samples_inputs), dim=1)

                            # Compute loss using the logits from the model copy
                            loss_samples = self.criterion(outputs_samples, samples_labels).clone().detach()

                            memory_buffer.update_loss(loss_samples, samples_labels)
                            # print('Loss updated at iteration:', minibatch)
                        model.train()
                        
                    # mean loss
                    # loss = loss.mean()
                    
                    # loss.item() # shape: torch.Size([256])

                    # Compute training statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.wandb:
                        wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                    # Print information every 50 minibatches
                    if minibatch % num_print == 0: 
                        print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                        
                        running_loss = 0.0
                    


                tmp_acc = self.validate(model, 'validation', task_id = task_id)
    

            return model, memory_buffer


    def ER_AAECB(self, 
                model, 
                memory_buffer, 
                task_id = None,):
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        # memory_buffer = memory_buffer.to(self.device)

        # Train model
        best_acc = 0

        if task_id == 'cil_task_0' or task_id == 'dil_task_0':

            if self.args.load_model:
                print('Training from pre-trained model')
                print('task_id', task_id)
                num_epochs = 1
                best_ep = -1
                flag_to_stop = False
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288

                    print("Epoch: " + str(epoch+1) +"/1")
                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    
                    total_time = 0
                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:
                            
                            memory_buffer.add_data(inputs, labels)
                            _, predicted = torch.max(outputs.data, 1)
                            TP_all = (predicted == labels) # True Positive size: torch.Size([256])
                            classes = labels.unique()
                            accuracy_per_class = {}
                            for c in classes:
                                TP = TP_all[labels == c]
                                accuracy_per_class[c.item()] = TP.float().mean().item()
                            memory_buffer.update_acc(accuracy_per_class)
                            


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer
            
            else :
                print('Training from scratch')
                print('task_id', task_id)
                num_epochs = 1 if self.args.debug else self.config['epochs']
                best_ep = -1
                flag_to_stop = False

                total_time = 0
                for epoch in range(0, num_epochs):

                    data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                    print("Data length: " + str(len(data))) # 288
                    
                    if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                    else:
                        print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                    model.train()
                    self.scheduler.step()

                    running_loss = 0.0
                    total = 0
                    correct = 0   

                    # num_iter = 20 if self.training_parameters['debug'] else len(data)
                    num_iter = len(data)
                    num_print = 20
                    

                    for minibatch in range(num_iter): 
                    # for minibatch in range(1): 

                        inputs, labels, _ = data[0]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                        # labels = torch.Tensor(labels).to(self.device).long()
                        inputs.unsqueeze_(1)
                        if inputs.size(0) == 0:
                            break
                        self.optimizer.zero_grad()

                        # Train, compute loss, update optimizer
                        model = model.to(self.device)
                        outputs = F.softmax(model(inputs), dim=1)
                        loss = self.criterion(outputs, labels) # torch.Size([256])

                        temp_loss = loss.clone().detach() # torch.Size([256])
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()

                        if epoch == num_epochs - 1 or flag_to_stop:

                            memory_buffer.add_data(inputs, labels)
                            _, predicted = torch.max(outputs.data, 1)
                            TP_all = (predicted == labels) # True Positive size: torch.Size([256])
                            classes = labels.unique()
                            accuracy_per_class = {}
                            for c in classes:
                                TP = TP_all[labels == c]
                                accuracy_per_class[c.item()] = TP.float().mean().item()
                            memory_buffer.update_acc(accuracy_per_class)


                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if self.args.wandb:
                            wandb.log({'Pre-train loss': loss.item(), 'accuracy': 100 * correct / total})   

                        # Print information every 50 minibatches
                        if minibatch % num_print == 0: 
                            print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                            running_loss = 0.0


                    tmp_acc = self.validate(model, 'validation', task_id = task_id)
                    if self.args.early_stop:
                        if flag_to_stop == True:
                            print('Early stop at epoch: ', epoch)
                            break
                        # Save best performing network
                        if (tmp_acc > best_acc):
                            best_acc = tmp_acc
                            best_ep  = epoch
                            best_state_dict = model.state_dict()

                        patience = self.config['patience']
                        
                        if (epoch >= best_ep + patience):
                            flag_to_stop = True
                            
                    else:
                        best_state_dict = model.state_dict()
            

                timestr = time.strftime("%Y%m%d-%H%M%S")
                if self.args.early_stop:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                else:
                    model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '.pth'
                model_path = './models/' + model_name

                torch.save(best_state_dict, model_path)
                print('model saved at: ', model_path)

                return model_path, memory_buffer

        else:
            num_epochs = 1 if self.args.debug else self.config['epochs_CL']
   
            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0' or task_id == 'dil_task_0':
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs']))
                else:
                    print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

                model.train()
                self.scheduler.step()

                running_loss = 0.0
                total = 0
                correct = 0   

                # num_iter = 20 if self.training_parameters['debug'] else len(data)
                num_iter = len(data)
                num_print = 20

                for minibatch in range(num_iter): 

                    inputs, labels, _ = data[0]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    # labels = torch.Tensor(labels).to(self.device).long()
                    inputs.unsqueeze_(1)
                    if inputs.size(0) == 0:
                        break
                    input_size = int(inputs.size(0))
                    # update memory buffer in the last epoch (to ensure that the memory buffer
                    # is not updated multiple times)
                    
                    samples_inputs, samples_labels = memory_buffer.get_data(input_size) 
    
                    # samples_inputs = samples_inputs.to(self.device)
                    # samples_labels = samples_labels.to(self.device).long()
                    samples_labels = samples_labels.long()
                    # samples_labels = torch.Tensor(samples_labels).to(self.device).long()
                    inputs = torch.cat((inputs, samples_inputs), 0) # torch.Size([256, 1, 49, 10])
                    labels = torch.cat((labels, samples_labels), 0) # torch.Size([256])
                    
                    # Zero out the parameter gradients after each mini-batch
                    self.optimizer.zero_grad()

                    # Train, compute loss, update optimizer
                    model = model.to(self.device)
                    outputs = F.softmax(model(inputs), dim=1)
                    loss = self.criterion(outputs, labels) # torch.Size([256])

                    loss_samples = loss[input_size:].clone().detach()
                    loss_new = loss[:input_size].clone().detach()

                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()
                    
                    if epoch == num_epochs - 1:
                    # if epoch == 0:
                        # add only first 128 input, loss and labels
                        memory_buffer.add_data(inputs[:input_size], labels[:input_size])
                        _, predicted = torch.max(outputs.data, 1)
                        TP_all = (predicted == labels) # True Positive size: torch.Size([256])
                        classes = labels.unique()
                        accuracy_per_class = {}
                        for c in classes:
                            TP = TP_all[labels == c]
                            accuracy_per_class[c.item()] = TP.float().mean().item()
                        memory_buffer.update_acc(accuracy_per_class)
                        
                    # mean loss
                    # loss = loss.mean()
                    
                    # loss.item() # shape: torch.Size([256])

                    # Compute training statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.wandb:
                        wandb.log({'training loss': loss.item(), 'accuracy': 100 * correct / total})   

                    # Print information every 50 minibatches
                    if minibatch % num_print == 0: 
                        print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / num_print, 100 * correct / total))
                        
                        running_loss = 0.0
                    


                tmp_acc = self.validate(model, 'validation', task_id = task_id)
    

            return model, memory_buffer


