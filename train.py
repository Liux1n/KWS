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


import dataset
import torch
import wandb
from typing import Optional
from utils import confusion_matrix, npy_to_txt, per_noise_accuracy, load_config

import torch.nn.functional as F



class Train():

    def __init__(self, audio_processor, training_parameters, model, device, args, config):
        self.audio_processor = audio_processor
        self.training_parameters = training_parameters
        self.model = model
        self.device = device
        self.args = args
        self.config = config

        # Training hyperparameters
        self.criterion = torch.nn.CrossEntropyLoss()
        intitial_lr = 0.001
        self.optimizer = torch.optim.Adam(model.parameters(), lr = intitial_lr)
        lambda_lr = lambda epoch: 1 if epoch<15 else 1/5 if epoch < 25 else 1/10 if epoch<35 else 1/20
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)


    def validate(self, model = None, mode='validation', batch_size = 0, 
                 statistics = False, integer = False, save = False, task = None):
        # Validate model

        training_parameters = self.training_parameters
        if (batch_size != 0):
            training_parameters['batch_size'] = batch_size   


        data = dataset.AudioGenerator(mode, self.audio_processor, training_parameters, task = task)
        model.eval()  

        correct = 0
        total = 0

        with torch.no_grad():

            if batch_size == -1:
                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).long().to(self.device)
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

                if statistics == True:
                    # conf_matrix(labels, predicted, self.training_parameters)
                    per_noise_accuracy(labels, predicted, noises)

            else:

                for minibatch in range(len(data)):
                    inputs, labels, noises = data[0]
                    inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    labels = torch.Tensor(labels).long().to(self.device)
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
                        print('[%3d / %3d] accuracy: %.3f' % (minibatch + 1, len(data),  100 * correct / total))
                        
                        running_loss = 0.0

                    if statistics == True:
                        # conf_matrix(labels, predicted, self.training_parameters)
                        per_noise_accuracy(labels, predicted, noises)

        print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        if self.args.wandb:
            wandb.log({'val_accuracy': 100 * correct / total})
        return(100 * correct / total)


    def train(self, model): # Train from scratch
    
        
        best_acc = 0

        num_epochs = 1 if self.args.debug else self.training_parameters['epochs']
        for epoch in range(0, num_epochs):

            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, self.args.task, task = None)
            print("Data length: " + str(len(data)))
            # take only 10 minibatches
            # data = data[:10]
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            num_iter = 20 if self.training_parameters['debug'] else len(data)

            for minibatch in range(num_iter): 
            

                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).to(self.device).long()

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

                # Print information every 20 minibatches
                if minibatch % 20 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                    if self.args.wandb:
                        wandb.log({'loss': running_loss / 10, 'accuracy': 100 * correct / total})   
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation')

        model_name = self.args.method + '_12_' + self.args.task + '_model.pth'
        PATH = './models/' + model_name

        torch.save(model.state_dict(), PATH)


    def continual_train(self, model):
        
        # Train model
        best_acc = 0
        num_epochs = 1 if self.args.debug else self.config['epochs_CL']
        # memory_buffer_size = self.config['memory_buffer_size']
        
        for epoch in range(0, num_epochs):

            print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, self.args.task, task = None)
            print("Data length: " + str(len(data))) # 288


            if self.args.method == 'finetune': # freeze the first layers
                
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
                    inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    labels = torch.Tensor(labels).to(self.device).long()

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

                    # Print information every 20 minibatches
                    if minibatch % 20 == 0: 
                        print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                        if self.args.wandb:
                            wandb.log({'loss': running_loss / 10, 'accuracy': 100 * correct / total})   
                        running_loss = 0.0

                tmp_acc = self.validate(model, 'validation')



        model_name = self.args.method + '_' + self.args.task + '_model.pth'
        PATH = './models/' + model_name
        torch.save(model.state_dict(), PATH)


    def er_random_train(self, 
                        model, 
                        memory_buffer, 
                        base = False):
    

        memory_buffer_size = self.config['memory_buffer_size']
        # Train model
        best_acc = 0
        num_epochs = 1 if self.args.debug else self.config['epochs_CL']

  
        for epoch in range(0, num_epochs):

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, self.args.task, task = None)

            if epoch == 0:

                if base:

                    #target_words='yes,no,up,down,left,right,on,off,stop,go,'
                    #2  ,3 ,4 ,5   ,6   ,7    ,8 ,9  ,10  ,11
                    
                    # data = tuple(tensor.to('cpu') if torch.is_tensor(tensor) else tensor for tensor in data)
    
                    print('initializing memory buffer')
                    for minibatch in range(int(memory_buffer_size/128)):
                        # return a random batch of data with batch size 128.
                        inputs_mb, labels_mb, noises_mb = data[0]
                        inputs_mb = torch.Tensor(inputs_mb[:,None,:,:]).to(self.device) # ([128, 1, 49, 10])
                        labels_mb = torch.Tensor(labels_mb).to(self.device).long() # ([128])
                        memory_buffer['inputs'].append(inputs_mb)
                        memory_buffer['labels'].append(labels_mb)
                        
                    # concatenate all the minibatches
                    memory_buffer['inputs'] = torch.cat(memory_buffer['inputs'], 0)
                    memory_buffer['labels'] = torch.cat(memory_buffer['labels'], 0)

                    print('memory buffer input shape: ' + str(memory_buffer['inputs'].shape)) # torch.Size([1024, 1, 49, 10])
                    print('memory buffer label shape: ' + str(memory_buffer['labels'].shape)) # torch.Size([1024])
                else:
                    print('Loading memory buffer')
                    memory_buffer = memory_buffer
                    # convert to tensor
                    memory_buffer['inputs'] = torch.Tensor(memory_buffer['inputs']).to(self.device)
                    memory_buffer['labels'] = torch.Tensor(memory_buffer['labels']).to(self.device).long()


            # data = tuple(tensor.to('cpu') if torch.is_tensor(tensor) else tensor for tensor in data)
            print("Data length: " + str(len(data))) # 288
            
            print("Epoch: " + str(epoch+1) +"/" + str(self.config['epochs_CL']))

            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            # num_iter = 20 if self.training_parameters['debug'] else len(data)
            num_iter = len(data)

            for minibatch in range(num_iter): 

                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).to(self.device).long()
                # print("inputs shape: " + str(inputs.shape)) # ([128, 1, 49, 10])
                # print("labels shape: " + str(labels.shape)) # ([128])
                
                ######################################################
                ## Experience Replay with COMPLETELY random samples:##
                ######################################################
                # memory_buffer['input']: # torch.Size([1024, 1, 49, 10])
                # memory_buffer['labels']: # torch.Size([1024])

                # concatenate the current minibatch with the memory buffer
                candidate_input = torch.cat((inputs, memory_buffer['inputs']), 0) # torch.Size([1024+128, 1, 49, 10])
                candidate_label = torch.cat((labels, memory_buffer['labels']), 0) # torch.Size([1024+128])

                # randomly select 1024 samples from the candidate_input as the new memory buffer
                idx_new = torch.randperm(candidate_input.size(0))[:memory_buffer_size]
                memory_buffer['inputs'] = candidate_input[idx_new]
                memory_buffer['labels'] = candidate_label[idx_new]

                # randomly select 128 samples from the memory buffer
                idx = torch.randperm(memory_buffer['inputs'].size(0))[:128]
                inputs_er = memory_buffer['inputs'][idx]
                labels_er = memory_buffer['labels'][idx]

                # concatenate the current minibatch with the experience replay samples
                inputs = torch.cat((inputs, inputs_er), 0)
                labels = torch.cat((labels, labels_er), 0)

                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                # model = model.to(self.device)
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
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                    if self.args.wandb:
                        wandb.log({'loss': running_loss / 10, 'accuracy': 100 * correct / total})   
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation')

        

        model_name = self.args.method + '_' + self.args.task + '_model.pth'
        PATH = './models/' + model_name
        torch.save(model.state_dict(), PATH)
        return memory_buffer



    def adapt (self, model, noise):

        self.training_parameters['noise_test'] = noise
        self.training_parameters['noise_train'] = noise

         # Train model
        best_acc = 0
        best_ep  = 0
        for epoch in range(0, self.training_parameters['epochs']):

            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

            # task = -1      - Silent training
            # task = None    - Mixed training
            # task = epoch-1 - NaiveCL
            # task = n       - Fixed-noise training

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task = None)
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            for minibatch in range(len(data)):

                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).to(self.device).long()

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

                # Print information every 20 minibatches
                if minibatch % 20 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation', task = 0)

            # Save best performing network
            if (tmp_acc > best_acc):
                best_acc = tmp_acc
                best_ep  = epoch
                BEST_PATH = './model_ep' + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                torch.save(model.state_dict(), BEST_PATH)

            patience = 10
            if (epoch >= best_ep + patience):
              break


        PATH = './model.pth'
        torch.save(model.state_dict(), PATH)

        
