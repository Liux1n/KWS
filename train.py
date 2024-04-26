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

            if task_id == 'cil_task_0':
                num_iter = int((17/35)*num_iter)
            elif task_id == 'cil_task_1' or task_id == 'cil_task_2' or task_id == 'cil_task_3':
                # num_iter = int((1/35)*num_iter)
                num_iter = int((6/35)*num_iter)
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
                    # get current time
                    # import time
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    # model_name = self.args.method + '_' + timestr + '_' + self.args.task + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                    # PATH = './models/' + model_name
                    # #BEST_PATH = './model_ep' + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                    # torch.save(model.state_dict(), PATH)
                    best_state_dict = model.state_dict()

                patience = 10
                if (epoch >= best_ep + patience):
                    break
            else:
                best_state_dict = model.state_dict()
        

        timestr = time.strftime("%Y%m%d-%H%M%S")

        model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
        PATH = './models/' + model_name

        torch.save(best_state_dict, PATH)


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
            if task_id == 'cil_task_0':
                num_iter = int((17/35)*num_iter)
            elif task_id == 'cil_task_1' or task_id == 'cil_task_2' or task_id == 'cil_task_3':
                num_iter = int((6/35)*num_iter)

            # elif task_id == 'dil_task_1' or task_id == 'dil_task_2' or task_id == 'dil_task_3':
            #     num_iter = int((3/18)*num_iter)

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

                # Print information every 50 minibatches
                if minibatch % 50 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, num_iter, running_loss / 10, 100 * correct / total))
                    if self.args.wandb:
                        wandb.log({'loss': running_loss / 10, 'accuracy': 100 * correct / total})   
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation', task_id = task_id)

        return model
    
    def ER_NRS(self, 
                model, 
                memory_buffer, 
                task_id = None,):
    
        # memory_buffer = memory_buffer.to(self.device)
        memory_buffer_size = self.config['memory_buffer_size']
        # Train model
        best_acc = 0
        num_epochs = 1 if self.args.debug else self.config['epochs_CL']
        # memory_buffer.to_device(self.device)
  
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
            if task_id == 'cil_task_0':
                num_iter = int((17/35)*num_iter)
            elif task_id == 'cil_task_1' or task_id == 'cil_task_2' or task_id == 'cil_task_3':
                num_iter = int((6/35)*num_iter)
            
            # elif task_id == 'dil_task_1' or task_id == 'dil_task_2' or task_id == 'dil_task_3':
            #     num_iter = int((3/18)*num_iter)
            num_print = 50

            for minibatch in range(num_iter): 

                inputs, labels, _ = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).to(self.device).long()
                
                # update memory buffer in the last epoch (to ensure that the memory buffer
                # is not updated multiple times)
                if epoch == num_epochs - 1:
                    memory_buffer.add_data(inputs, labels)

                samples_inputs, samples_labels = memory_buffer.get_data()
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
        memory_buffer_size = self.config['memory_buffer_size']
        # Train model
        best_acc = 0
        if task_id == 'cil_task_0':
            num_epochs = self.config['epochs']
            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0':
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
                if task_id == 'cil_task_0':
                    num_iter = int((17/35)*num_iter)
                elif task_id == 'cil_task_1' or task_id == 'cil_task_2' or task_id == 'cil_task_3':
                    num_iter = int((6/35)*num_iter)
                
                # elif task_id == 'dil_task_1' or task_id == 'dil_task_2' or task_id == 'dil_task_3':
                #     num_iter = int((3/18)*num_iter)
                num_print = 50
                

                for minibatch in range(num_iter): 
                # for minibatch in range(1): 

                    inputs, labels, _ = data[0]
                    inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    labels = torch.Tensor(labels).to(self.device).long()
                    self.optimizer.zero_grad()

                    # Train, compute loss, update optimizer
                    model = model.to(self.device)
                    outputs = F.softmax(model(inputs), dim=1)
                    loss = self.criterion(outputs, labels) # torch.Size([256])
                    memory_buffer.update_loss(loss)

                    if epoch == num_epochs - 1:
                        # update loss
                        
                        memory_buffer.add_data(inputs, loss, labels)
                        

                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()


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

                    patience = 10
                    if (epoch >= best_ep + patience):
                        break
                else:
                    best_state_dict = model.state_dict()
        

            timestr = time.strftime("%Y%m%d-%H%M%S")
            if self.args.early_stop:
                model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + str(best_ep) + '_acc' + str(best_acc) + '.pth'
            else:
                model_name = model_name = self.args.mode + '_' + self.args.method + '_' + timestr + '_acc' + str(best_acc) + '.pth'
            PATH = './models/' + model_name

            torch.save(best_state_dict, PATH)

        else:
            num_epochs = 1 if self.args.debug else self.config['epochs_CL']
            # num_epochs = 2
   

            for epoch in range(0, num_epochs):

                data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task_id, task = None)
                print("Data length: " + str(len(data))) # 288
                
                if task_id == 'cil_task_0':
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
                if task_id == 'cil_task_0':
                    num_iter = int((17/35)*num_iter)
                elif task_id == 'cil_task_1' or task_id == 'cil_task_2' or task_id == 'cil_task_3':
                    num_iter = int((6/35)*num_iter)
                
                # elif task_id == 'dil_task_1' or task_id == 'dil_task_2' or task_id == 'dil_task_3':
                #     num_iter = int((3/18)*num_iter)
                num_print = 50

                for minibatch in range(num_iter): 

                    inputs, labels, _ = data[0]
                    inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    labels = torch.Tensor(labels).to(self.device).long()
                    
                    # update memory buffer in the last epoch (to ensure that the memory buffer
                    # is not updated multiple times)
                    
                    samples_inputs, samples_labels = memory_buffer.get_data() # 
    
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
                    memory_buffer.update_loss(loss[128:])
                    
                    # if epoch == num_epochs - 1:
                    if epoch == 0:
                        
                        # add only first 128 input, loss and labels
                        memory_buffer.add_data(inputs[:128], loss[:128], labels[:128])
                        # update loss
                        

                    # mean loss
                    # loss = loss.mean()
                    loss.mean().backward()
                    self.optimizer.step()
                    # loss.item() # shape: torch.Size([256])
        
                    

                    # Compute training statistics
                    mean_loss = torch.mean(loss)
                    running_loss += mean_loss
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


    # def adapt (self, model, noise):

    #     self.training_parameters['noise_test'] = noise
    #     self.training_parameters['noise_train'] = noise

    #      # Train model
    #     best_acc = 0
    #     best_ep  = 0
    #     for epoch in range(0, self.training_parameters['epochs']):

    #         print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

    #         # task = -1      - Silent training
    #         # task = None    - Mixed training
    #         # task = epoch-1 - NaiveCL
    #         # task = n       - Fixed-noise training

    #         data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task = None)
    #         model.train()
    #         self.scheduler.step()

    #         running_loss = 0.0
    #         total = 0
    #         correct = 0   

    #         for minibatch in range(len(data)):

    #             inputs, labels, noises = data[0]
    #             inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
    #             labels = torch.Tensor(labels).to(self.device).long()

    #             # Zero out the parameter gradients after each mini-batch
    #             self.optimizer.zero_grad()

    #             # Train, compute loss, update optimizer
    #             model = model.to(self.device)
    #             outputs = F.softmax(model(inputs), dim=1)
    #             loss = self.criterion(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()

    #             # Compute training statistics
    #             running_loss += loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #             # Print information every 20 minibatches
    #             if minibatch % 20 == 0: 
    #                 print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
    #                 running_loss = 0.0

    #         tmp_acc = self.validate(model, 'validation', task = 0)

    #         # Save best performing network
    #         if (tmp_acc > best_acc):
    #             best_acc = tmp_acc
    #             best_ep  = epoch
    #             BEST_PATH = './model_ep' + str(best_ep) + '_acc' + str(best_acc) + '.pth'
    #             torch.save(model.state_dict(), BEST_PATH)

    #         patience = 10
    #         if (epoch >= best_ep + patience):
    #           break


    #     PATH = './model.pth'
    #     torch.save(model.state_dict(), PATH)

        
