#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 07:46:29 2023

@author: whm
"""

import torch
import torchvision
import numpy as np

from models import CNNMnist, CNNCifar
from options import args_parser


import os
import copy
import time
import pickle

from torchvision import datasets, transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference,test_inference_class1
from utils import get_dataset, average_weights, exp_details
import inversefed

#global parameter
GI_iteration=10000
problematic_class=5
delay=40
dc=True
delay_training=True
num_image_rec=128
switch=10000

experiment_id=2
decreased_update_fre=False

args = args_parser()
args.dataset='cifar'
global_model=CNNCifar(args=args)

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')


exp_details(args)

device = 'cuda'

# load dataset and user groups
mnist_data_dir = '../data/mnist/'
mnist_apply_transform = transforms.Compose(
    [torchvision.transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Resize((32,32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

mnist_train_dataset = datasets.MNIST(mnist_data_dir, train=True, download=True,
                               transform=mnist_apply_transform)

mnist_test_dataset = datasets.MNIST(mnist_data_dir, train=False, download=True,
                              transform=mnist_apply_transform)
mnist_labels = np.array(mnist_train_dataset.targets)
mnist_idxs = np.arange(60000)
mnist_idxs_labels = np.vstack((mnist_idxs, mnist_labels))
mnist_idxs_labels = mnist_idxs_labels[:, mnist_idxs_labels[1, :].argsort()]
mnist_idxs = mnist_idxs_labels[0, :]




svhn_data_dir = '../data/svhn/'
svhn_apply_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

svhn_train_dataset = datasets.SVHN(svhn_data_dir, split='train', download=True,
                               transform=svhn_apply_transform)

svhn_test_dataset = datasets.SVHN(svhn_data_dir, split='test', download=True,
                              transform=svhn_apply_transform)
svhn_labels = np.array(svhn_train_dataset.labels)
svhn_idxs = np.arange(73257)
svhn_idxs_labels = np.vstack((svhn_idxs, svhn_labels))
svhn_idxs_labels = svhn_idxs_labels[:, svhn_idxs_labels[1, :].argsort()]
svhn_idxs = svhn_idxs_labels[0, :]



def cal_reselected_sample(num_sample_per_class,idxs,ini=0):
    start_sample=[(num_sample_per_class[0]),sum(num_sample_per_class[0:2]),sum(num_sample_per_class[0:3]),sum(num_sample_per_class[0:4])\
                  ,sum(num_sample_per_class[0:5]),sum(num_sample_per_class[0:6]),sum(num_sample_per_class[0:7])\
                      ,sum(num_sample_per_class[0:8]),sum(num_sample_per_class[0:9]),sum(num_sample_per_class[0:10])]

    reselected_idxs =[]
    for i in range(len(start_sample)):
        reselected_idxs=np.concatenate([reselected_idxs,idxs[start_sample[i]+ini:start_sample[i]+4500+ini]])
    return reselected_idxs

merged_train_dataset=mnist_train_dataset+svhn_train_dataset
merged_test_dataset=mnist_test_dataset+svhn_test_dataset
mergerd_idxs=np.concatenate((mnist_idxs,svhn_idxs+60000))
mnist_reselected_idxs=cal_reselected_sample([0,5923,6742,5958,6131,5842,5421,5918,6265,5851,5949],mergerd_idxs)
svhn_reselected_idxs=cal_reselected_sample([0,4948,13861,10585,8497,7458,6882,5727,5595,5045,4659],mergerd_idxs,ini=60000)


num_users=100    
num_client_data=int(45000/100)
start_idx=range(0,45000-1,num_client_data)
np.random.seed(1)
start_idx=np.random.choice(start_idx,num_users,replace=False)


global_model.to(device)
global_model.train()
print(global_model)

global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_pre, counter = 0, 0
l_test_acc=[]
l_test_loss = []
l_test_acc_class1=[]
pre_index=0
l_test_acc_mnist=[]
l_test_acc_svhn=[]

l_test_acc_class1_mnist=[]
l_test_acc_class1_svhn=[]
weights_history=[]

l_cnt=[]

np.random.seed(4);
random_drop=np.random.rand(500)
cnt_drop=0

for epoch in tqdm(range(args.epochs)):
    
    dict_users = {i: np.array([]) for i in range(num_users)}
    percent=epoch/args.epochs
    for i in range(num_users):
        client_mnist_idxs=mnist_reselected_idxs[start_idx[i]+int(percent*num_client_data):start_idx[i]+num_client_data]
        client_svhn_idxs=svhn_reselected_idxs[start_idx[i]:start_idx[i]+int(num_client_data*(percent))]
        dict_users[i]=np.concatenate((client_mnist_idxs, client_svhn_idxs))
    user_groups=dict_users

    if epoch<delay:
        weights_history.append(global_weights)
        
    if epoch>delay-1:
        for i in range(len(weights_history)-1):
            weights_history[i]=weights_history[i+1]
        weights_history[len(weights_history)-1]=global_weights
        

    local_weights, local_losses = [], []


    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    if pre_index+m>args.num_users:
        pre_index=0
    idxs_users=range(pre_index,pre_index+m,1)
    pre_index=pre_index+m
    

    
    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=merged_train_dataset,
                                  idxs=user_groups[idx], logger=logger)
        delete_update=0
        if  (merged_train_dataset[int(next(iter(user_groups[idx])))][1] ==problematic_class  )and epoch>delay-1 and delay_training:
            cnt_drop=cnt_drop+1

            print('delay')
            w0=weights_history[0]
            global_model.load_state_dict(w0)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            
            delta_w={}
            for key, value in w.items():
                delta_w[key]=w[key]-w0[key]
            
            if dc :
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
            else:
                
                dc_w={}
                for key, value in w.items():
                    dc_w[key]=delta_w[key].detach()+global_weights[key]
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
        local_model = LocalUpdate(args=args, dataset=merged_train_dataset,
                                  idxs=user_groups[idx], logger=logger)
        acc, loss = local_model.inference(model=global_model)
        list_acc.append(acc)
        list_loss.append(loss)
    train_accuracy.append(sum(list_acc)/len(list_acc))


    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, merged_test_dataset)
    l_test_acc.append(test_acc)
    test_acc, test_loss = test_inference(args, global_model, mnist_test_dataset)
    l_test_acc_mnist.append(test_acc)
    test_acc, test_loss = test_inference(args, global_model, svhn_test_dataset)
    l_test_acc_svhn.append(test_acc)

    print(f' \n test acc:{test_acc} ')
    test_acc_class1= test_inference_class1(args, global_model, merged_test_dataset,problematic_class)
    l_test_acc_class1.append(test_acc_class1)
    test_acc_class1_mnist= test_inference_class1(args, global_model, mnist_test_dataset,problematic_class)
    l_test_acc_class1_mnist.append(test_acc_class1_mnist)
    test_acc_class1_svhn= test_inference_class1(args, global_model, svhn_test_dataset,problematic_class)
    l_test_acc_class1_svhn.append(test_acc_class1_svhn)
    print(f' \n class 0 test acc :{test_acc_class1} ')
    print(f' \n Results after {epoch} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    
    
    if epoch%25==1:
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
    
















