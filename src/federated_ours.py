#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import date, timedelta

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, hourly_data
from networkx.algorithms import community


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    print(args)
    mod = args.modularity
    centralized = args.central
    # sys.exit()

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

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
    
    # initializing energy usage and cluster lists for graphs
    clusters = []
    avg_battery_round = []
    energy_node, energy_cloud, energy_tot, energy_rec, num_sel_users = 0.02, 0.27, 10, 0, 3
    if centralized:
        energy_cloud = 0.21
    used_energy = np.zeros(args.num_users)

    # data and how to iterate through data
    data = pd.read_csv('../data/Top_250_Clients_9_to_5.csv')
    today = date(2020, 5, 1)
    deltadate = timedelta(days=1)
    hour = 9

    for epoch in tqdm(range(args.epochs)):
        chosen_nodes = np.array([])
        all_non_chosen_nodes = []

        # data is from 9 AM to 5 PM UTC
        hour = 9 + epoch % 9

        # get data for a comm round
        df_commround = data[data['utc_date'] == str(today)]
        df_commround = df_commround[df_commround['utc_hour'] == hour]
        G = hourly_data(df_commround)

        global_model.train()
        current_nodes = np.array(G.nodes)

        if np.size(current_nodes) == 0:
            # update time and date for next round if needed
            # data is from 9 AM to 5 PM UTC
            hour += (epoch + 1) % 9
            if (epoch + 1) % 9 == 0:
                today = today + deltadate
            
            used_energy = np.maximum(used_energy - energy_rec, 0)
            train_accuracy.append(train_accuracy[-1])
            train_loss.append(train_loss[-1])
            avg_battery_round.append(avg_battery_round[-1])
            continue

        idx_nodes = np.where(used_energy[current_nodes] <= energy_tot - energy_cloud)[0]
        eligible_nodes = current_nodes[idx_nodes]

        if args.modularity and G.edges:
            clusters = community.greedy_modularity_communities(G)
            clusters = [list(x) for x in clusters]

            # Step 1: For each cluster, choose the highest degree node with cnt <= M - n
            for cluster in clusters:
                # Sort cluster nodes by degree in descending order
                sorted_nodes = sorted(cluster, key=lambda node: G.degree[node], reverse=True)
                
                # Find the first node which satisfies energy constraints
                chosen_node = next((node for node in sorted_nodes if used_energy[node] <= energy_tot - energy_node), None)
                if chosen_node:
                    chosen_nodes = np.append(chosen_nodes, int(chosen_node))
        else:
            if centralized:
                chosen_nodes = eligible_nodes
            else:
                chosen_nodes = np.random.choice(eligible_nodes, min(num_sel_users, len(eligible_nodes)))#, replace=False)

        chosen_nodes = chosen_nodes.astype(int)

        # Step 2: Update used_energy for chosen nodes
        if chosen_nodes.size == 0:
            used_energy = np.maximum(used_energy - energy_rec, 0)
            train_accuracy.append(train_accuracy[-1])
            train_loss.append(train_loss[-1])
            avg_battery_round.append(np.mean(used_energy))
            
        else:
            used_energy[chosen_nodes] += energy_cloud
            round_energy = np.array([])
            round_energy = np.append(round_energy, np.ones(len(chosen_nodes))*energy_cloud)

            local_weights, local_losses = [], []

            # include chosen node as part of aggregation
            for idx in chosen_nodes:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            if not centralized:
                for idx in chosen_nodes:
                    neighbors = G.neighbors(idx)
                    eligible_neighbors = [neigh for neigh in neighbors if used_energy[neigh] <= energy_tot - energy_node]
                    used_energy[eligible_neighbors] += energy_node
                    round_energy = np.append(round_energy, np.ones(len(eligible_neighbors))*energy_node)

                    non_chosen_non_neighbor_nodes = [j for j in range(args.num_users) 
                        if j not in chosen_nodes and j not in neighbors]
                    all_non_chosen_nodes.extend(non_chosen_non_neighbor_nodes)

                    for neigh in eligible_neighbors:
                        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                                idxs=user_groups[neigh], logger=logger)
                        w, loss = local_model.update_weights(
                            model=copy.deepcopy(global_model), global_round=epoch)
                        local_weights.append(copy.deepcopy(w))
                        local_losses.append(copy.deepcopy(loss))
            else:
                non_chosen_nodes = [j for j in range(args.num_users) if j not in chosen_nodes]
                all_non_chosen_nodes.extend(non_chosen_nodes)

            # Update used_energy for nodes that are neither chosen nor neighbors of chosen nodes
            all_non_chosen_nodes = np.unique(all_non_chosen_nodes)  # Remove duplicates
            used_energy[all_non_chosen_nodes] -= energy_rec

            # Step 5: Ensure used_energy values are non-negative
            used_energy = np.maximum(0, used_energy)

            # update global weights
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
            avg_battery_round.append(np.mean(used_energy))


            #print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        
        # update time and date for next round if needed
        # data is from 9 AM to 5 PM UTC
        hour += (epoch + 1) % 9
        if (epoch + 1) % 9 == 0:
            today = today + deltadate


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
    np.savez('../save/fed_{}_{}_{}_iid[{}]_M[{}]_Central[{}]'.
                format(args.dataset, args.model, args.epochs,
                       args.iid, args.modularity, centralized), 
                       battery = (np.array(avg_battery_round)/energy_tot) * 100, 
                       train_accuracy = train_accuracy,
                       test_acc = test_acc)

    #Plot Loss curve
    # title = "Random Cluster Init"
    # if args.modularity:
    #     title = "Modularity Cluster Init"
    # elif centralized:
    #     title = "Centralized"

    # plt.figure()
    # plt.title('Battery Use vs Communication Round - ' + title)
    # plt.plot(range(len(avg_battery_round)), (np.array(avg_battery_round)/energy_tot) * 100, color='r')
    # plt.ylabel('Battery Percentage Used')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_iid[{}]_M[{}]_Central[{}]_battery.png'.
    #             format(args.dataset, args.model, args.epochs,
    #                    args.iid, args.modularity, centralized))
    
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication Rounds - ' + title)
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_iid[{}]_M[{}]_Central[{}]_acc.png'.
                # format(args.dataset, args.model, args.epochs,
                #        args.iid, args.modularity, centralized))
