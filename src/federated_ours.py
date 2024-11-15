#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import os
import copy
import time
import pickle
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    print(args)
    # sys.exit()

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'
    device = 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    energy_node, energy_cloud, energy_tot, energy_rec, num_sel_users = 2, 6, 10, 1, 10
    used_energy = np.zeros(args.num_users)
    adj_matrix = sio.loadmat('../data/Adjacency.mat')
    adj_matrix = adj_matrix['A']

    for epoch in tqdm(range(args.epochs)):
        cluster_weights, cluster_losses = [], []
        all_non_chosen_nodes = []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        eligible_nodes = np.where(used_energy <= energy_tot - energy_cloud)[0]
        chosen_nodes = np.random.choice(eligible_nodes, num_sel_users, replace=False)

        # Step 2: Update used_energy for chosen nodes
        used_energy[chosen_nodes] += energy_cloud

        for idx in chosen_nodes:
            neighbors = np.where(adj_matrix[idx] > 0)[0]
            eligible_neighbors = [neigh for neigh in neighbors if used_energy[neigh] <= energy_tot - energy_node]
            used_energy[eligible_neighbors] += energy_node

            non_chosen_non_neighbor_nodes = [
            j for j in range(args.num_users) if j not in chosen_nodes and j not in neighbors
            ]
            all_non_chosen_nodes.extend(non_chosen_non_neighbor_nodes)

            local_weights, local_losses = [], []

            # include chosen node as part of aggregation
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            for neigh in eligible_neighbors:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[neigh], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            cluster_weights.append(average_weights(local_weights))
            cluster_losses.append(sum(local_losses)/len(local_losses))

        # Update used_energy for nodes that are neither chosen nor neighbors of chosen nodes
        all_non_chosen_nodes = np.unique(all_non_chosen_nodes)  # Remove duplicates
        used_energy[all_non_chosen_nodes] -= energy_rec

        # Step 5: Ensure used_energy values are non-negative
        used_energy = np.maximum(0, used_energy)

        # update global weights
        global_weights = average_weights(cluster_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(cluster_losses) / len(cluster_losses)
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

        # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication Rounds - IID')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                # format(args.dataset, args.model, args.epochs, args.frac,
                #        args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication Rounds - IID')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
