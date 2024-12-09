#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import networkx as nx
import sys
from collections import OrderedDict
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def hourly_data(df_commround):
    """Returns a graph of all nodes that are active for a communication round"""
    threshold = 0.00089977 # distance threshold from MOHAWK
    result = {}
    adj_list = []

    for i in range(len(df_commround)):
        for j in range(i + 1, len(df_commround)):
            x1, y1 = df_commround.iloc[i]['geolat'], df_commround.iloc[i]['geolong']
            x2, y2 = df_commround.iloc[j]['geolat'], df_commround.iloc[j]['geolong']

            # Calculate the Euclidean distance
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Compare the distance with the threshold
            if distance < threshold:
                m, n = df_commround.iloc[i]['persistentid'], df_commround.iloc[j]['persistentid']
                # Add to the result dictionary
                if m not in result:
                    result[m] = []
                if m!= n:
                    result[m].append(n)
                adj_list = {k: np.unique(v) for k, v in result.items() if v}

    if not adj_list:
        G = nx.Graph()
        G.add_nodes_from(df_commround.iloc[:]['persistentid'])
    else:
        G = nx.from_dict_of_lists(adj_list)
        
    return G

def average_weights(w):
    """
    Returns the average of the weights.
    """
    if len(w) > 1:
        w_avg = copy.deepcopy(w[0])

    else:
        w_avg = copy.deepcopy(w)
        while not isinstance(w_avg, OrderedDict):
            w_avg = w_avg[0]

    # average the weights across each layer
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
