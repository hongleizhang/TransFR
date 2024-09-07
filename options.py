#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse

import torch.cuda as cuda


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--global_rounds', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--client_frac', type=float, default=0.2,
                        help='the fraction of clients: C')
    parser.add_argument('--local_epochs', type=int, default=30,
                        help="the number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=2048,
                        help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--server_lr', type=float, default=1e-3,
                        help='learning rate')

    # model arguments
    parser.add_argument('--model', type=str, default='TransFR', help='model name')
    parser.add_argument('--latent_dim', type=int, default=32, help='embedding size for users and items')
    parser.add_argument('--num_negative_train', type=int, default=4, help='number of negative samples for training')
    parser.add_argument('--num_negative_test', type=int, default=999, help='number of negative samples for test')
    parser.add_argument('--layers', type=list, default=[312, 128, 64, 32], help='mlp model layers')
    parser.add_argument('--local_head', type=int, default=0, help='local training for personalized local head')
    parser.add_argument('--top_k', type=int, default=10, help='the number of recommendation list')
    parser.add_argument('--cold_nums', type=int, default=20, help='the number of cold start items in user perspective')
    parser.add_argument('--optimizer', type=str, default='Adam', help="type \
                        of optimizer")
    parser.add_argument('--weight_decay', type=float, default=0, help="weight decacy of parameters")

    # other arguments
    parser.add_argument('--dataset', type=str, default='Music-Book', help="name \
                        of dataset")
    parser.add_argument('--data_type', type=str, default='implicit', help="explicit or implicit")
    parser.add_argument('--training_type', type=str, default='federated', help="federated or centralized")
    parser.add_argument('--device', nargs='?', default='cuda' if cuda.is_available() else 'cpu',
                        help="Which device to run the model")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args
