#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:02:39 2023

@author: whm
"""

import torch
import numpy as np
import os
import copy
import time
import pickle
from tqdm import tqdm
from tensorboardX import SummaryWriter


from models import  CNNMnist, CNNCifar
from options import args_parser
from update import LocalUpdate, test_inference,test_inference_class1
from utils import get_dataset, average_weights
import inversefed

experiment_id=1
#global parameter
GI_iteration=40000
problematic_class=5
delay=40
# 1 GI estimation  2 direct aggregation 3 DC-sdgd 4 weighted 0 undelay
dc=1

#delay_training=True
save_model=False
lr_decay=False
num_image_rec=256

switch_max=300

switch_point_reached=False
args = args_parser()
args.dataset='mnist'
global_model=CNNMnist(args=args)

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')
device = 'cuda'
train_dataset, test_dataset, user_groups = get_dataset(args)
global_model.to(device)
global_model.train()
global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
l_test_acc=[]
l_test_loss = []
l_test_acc_class1=[]
pre_index=0
switch = args.epochs
weights_history=[]
np.random.seed(4);
random_drop=np.random.rand(500)

for epoch in tqdm(range(args.epochs)):

    if lr_decay==True:
        if epoch >500:
            args.lr=0.001
        if epoch >600:
            args.lr=0.0001
        if epoch >700:
            args.lr=0.00001

    #cnt=039
    if epoch<delay:
        weights_history.append(global_weights)
        
    if epoch>delay-1:
        for i in range(len(weights_history)-1):
            weights_history[i]=weights_history[i+1]
        weights_history[len(weights_history)-1]=global_weights
        

    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    if pre_index+m>args.num_users:
        pre_index=0
    idxs_users=range(pre_index,pre_index+m,1)
    pre_index=pre_index+m
    

    
    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger)

        if  (train_dataset[int(next(iter(user_groups[idx])))][1] ==problematic_class  )and epoch>delay-1 and dc!=0:

            print('delay')
            w0=weights_history[0]
            global_model.load_state_dict(w0)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            
            delta_w={}
            for key, value in w.items():
                delta_w[key]=w[key]-w0[key]
            
            
            if dc==1 and (epoch<switch_max and epoch<switch):
                
                model=copy.deepcopy(global_model)
                local_lr = args.lr
                local_steps = args.local_ep
                use_updates = True
                setup = inversefed.utils.system_startup()
                if args.dataset == 'mnist':
                    dm = torch.as_tensor([0.5], **setup)[:, None, None]
                    ds = torch.as_tensor([0.25], **setup)[:, None, None]
                    img_shape=(1,28,28)
                else:
                    dm = torch.as_tensor([0.5,0.5,0.5], **setup)[:, None, None]
                    ds = torch.as_tensor([0.25,0.25,0.25], **setup)[:, None, None]
                    img_shape=(3,32,32)
                model.zero_grad()
                input_parameters=[]
                for key, value in delta_w.items(): 
                    if key[-1]=='t' or key[-1]=='s':
                        input_parameters.append(value.detach())

                list(delta_w.values())
                config = dict(signed=True,
                              boxed=True,
                              cost_fn='l1',
                              indices='def',
                              weights='equal',
                              lr=1,
                              optim='adam',
                              restarts=1,
                              max_iterations=GI_iteration,
                              total_variation=0,
                              init='randn',
                              filter='none',
                              lr_decay=True,
                              scoring_choice='loss')
                
                rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, local_lr, config,
                                                             use_updates=use_updates, num_images=num_image_rec)
                output, stats,rec_labels = rec_machine.reconstruct(input_parameters, labels=None, img_shape=img_shape)
                global_model.load_state_dict(global_weights)
                model=copy.deepcopy(global_model)
                rec_w= inversefed.reconstruction_algorithms.loss_steps_od(model, output, rec_labels, 
                                                                        lr=local_lr, local_steps=local_steps,
                                                                                 use_updates=use_updates)
                dc_w={}
                for key, value in w.items():
                    dc_w[key]=rec_w[key].detach()+global_weights[key]
                    dc_w[key].detach()
                
                w=dc_w
                
                if switch_point_reached==False:
                    global_model.load_state_dict(global_weights)
                    w_true, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=epoch)
                    dc_w={}
                    for key, value in w.items():
                        dc_w[key]=delta_w[key].detach()+global_weights[key]
                        dc_w[key].detach()
                    w_undc=dc_w
                    
                    e1=0
                    e2=0
                    for key, value in w.items():
                        e1=e1+torch.norm(w_true[key]-w[key],p=1)
                        e2=e2+torch.norm(w_true[key]-w_undc[key],p=1)
                    if e1<e2:
                        switch_point_reached=True
                        switch=epoch+delay
                    
            if dc==1 and (epoch>=switch_max or epoch<switch ):
                dc_w={}
                for key, value in w.items():
                    dc_w[key]=delta_w[key].detach()+global_weights[key]
                    dc_w[key].detach()
                w=dc_w
            if dc==2:
                dc_w={}
                for key, value in w.items():
                    dc_w[key]=delta_w[key].detach()+global_weights[key]
                    dc_w[key].detach()
                w=dc_w
            if dc==3:
                dc_w={}
                lamda=4
                for key, value in w.items():
                    dc_w[key]=global_weights[key]+delta_w[key].detach()-lamda*delta_w[key].detach()*delta_w[key].detach()*(global_weights[key]-w0[key])
                    dc_w[key].detach()
                w=dc_w
            if dc==4:
                dc_w={}
                for key, value in w.items():
                    dc_w[key]=delta_w[key].detach()*np.exp2(-delay/20)+global_weights[key]
                    dc_w[key].detach()
                w=dc_w

                
                
                    
        else:
            global_model.load_state_dict(global_weights)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
        

        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

    global_weights = average_weights(local_weights)

    # update global weights
    global_model.load_state_dict(global_weights)
    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # Calculate avg training accuracy over all users at every epoch
    list_acc, list_loss = [], []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger)
        acc, loss = local_model.inference(model=global_model)
        list_acc.append(acc)
        list_loss.append(loss)
    train_accuracy.append(sum(list_acc)/len(list_acc))

    
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    l_test_acc.append(test_acc)
    l_test_loss.append(test_loss)
    
    test_acc_class1= test_inference_class1(args, global_model, test_dataset,problematic_class)
    l_test_acc_class1.append(test_acc_class1)
    print(f' \n class 0 test acc :{test_acc_class1} ')
    
    print(f' \n test acc:{test_acc} ')
    print(f' \n Results after {epoch} global rounds of training:')

    
    if save_model==True and epoch%20==1:
        file_name = 'save/model_h/{}weights_{}.pkl'.\
            format(experiment_id,epoch)

        with open(file_name, 'wb') as f:
            pickle.dump(weights_history, f)
            
    if epoch%5==1:       
        file_name = 'save/objects/{}class1_acc.pkl'.\
            format(experiment_id)

        with open(file_name, 'wb') as f:
            pickle.dump(l_test_acc_class1, f)
            
        file_name = 'save/objects/{}acc.pkl'.\
            format(experiment_id)

        with open(file_name, 'wb') as f:
            pickle.dump(l_test_acc, f)
    








