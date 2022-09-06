import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
import random
import copy
from collections import OrderedDict, defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from datasets import CIFAR10_truncated, CIFAR100_truncated, CharacterDataset
from math import sqrt

import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.optim as optim
import torchvision.utils as vutils
import time
import random

from config import params
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file
from constants import *
from datastore import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None):

    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)
        
        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]

        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))

                ## Balance
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])

                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]

                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]
                
                min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                min_size = min(min_size_train, min_size_test)

        for j in range(n_parties):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    elif partition == "iid-label100":
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeldir100":
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label],fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < 50_000]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i], index[index >= 50_000]-50000).astype(int)

    elif partition == "noniid-labeluni":
        if dataset == "cifar10":
            num = 2
        elif dataset == "cifar100":
            num = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'cifar10':
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def compute_accuracy_shakes(model, dataloader, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    global_loss = 0.
    global_metric = 0.
    n_samples = 0

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    with torch.no_grad():
        for tmp in dataloader:
            for x, y, indices in tmp:                
                x = x.to(device)
                y = y.to(device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred, _ = model(x)
                global_loss += criterion(y_pred, y).sum().item() / chunk_len
                _, predicted = torch.max(y_pred, 1)
                correct = (predicted == y).float()
                acc = correct.sum()
                global_metric += acc.item() / chunk_len

    if was_training:
        model.train()

    return global_metric, n_samples, global_loss/n_samples


def compute_accuracy_fedRod(model, p_head, dataloader, sample_per_class, args, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        p_head.eval()
        was_training = True

    if args.dataset == "cifar10":
        class_number = 10
    elif args.dataset == "cifar100":
        class_number = 100

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)

                rep = model.produce_feature(x)
                if args.model == 'cnn':
                    out_g = model.fc3(rep)
                elif args.model == 'vit':
                    out_g = model.mlp_head(rep)

                out_p = p_head(rep.detach())
                out = out_g.detach() + out_p
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)

                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()
        p_head.train()

    return correct, total, total_loss/batch_count


def compute_accuracy_fedproto(model, global_protos, dataloader, args, device='cpu'):
    was_training = False
    loss_mse = nn.MSELoss().to(device)
    if global_protos == None:
        return 0, 1, 0.1
    if model.training:
        model.eval()
        was_training = True

    if args.dataset == "cifar10":
        class_number = 10
    elif args.dataset == "cifar100":
        class_number = 100
        
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                rep = model.produce_feature(x)       
                out = float('inf') * torch.ones(target.shape[0], class_number).to(device)
                for i, r in enumerate(rep):
                    for j, pro in global_protos.items():
                        out[i, j] = loss_mse(r, pro)
                pred_label = torch.argmin(out, dim=1)
                loss = criterion(out, target)
                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count


def compute_accuracy_loss(model, dataloader, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:

            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)

                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]
                
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count


def compute_accuracy_loss_knn(model, train_dataloader, test_dataloader, datastore, embedding_dim, args, device="cpu"):

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)
    datastore.clear()
    n_samples = len(train_dataloader.dataset)
    total = len(test_dataloader.dataset)

    if type(test_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
        test_dataloader = [test_dataloader]

    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100

    with torch.no_grad():
        train_features = 0
        train_labels = 0

        ff = 0
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                t_feature = model.produce_feature(x).detach()
                out = model.mlp_head(t_feature)
                t_feature = t_feature.cpu().numpy()

                if ff == 0:
                    ff = 1
                    train_labels = target.data.cpu().numpy()
                    train_features = t_feature
                else:
                    train_labels = np.hstack((train_labels, target.data.cpu().numpy()))
                    train_features = np.vstack((train_features, t_feature))

        test_features = 0
        test_labels = 0
        test_outputs = 0
        ff = 0
        for tmp in test_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                t_feature = model.produce_feature(x).detach()
                out = model.mlp_head(t_feature)
                t_feature = t_feature.cpu().numpy()

                if ff == 0:
                    ff = 1
                    test_labels = target.data.cpu().numpy()
                    test_features = t_feature
                    test_outputs = F.softmax(out, dim=1).cpu().numpy()
                else:
                    test_labels = np.hstack((test_labels, target.data.cpu().numpy()))
                    test_features = np.vstack((test_features, t_feature))
                    test_outputs = np.vstack((test_outputs, F.softmax(out, dim=1).cpu().numpy()))

        datastore.build(train_features, train_labels)
        distances, indices = datastore.index.search(test_features, args.k_value)
        similarities = np.exp(-distances / (embedding_dim * 1.))
        neighbors_labels = datastore.labels[indices]
        masks = np.zeros(((n_classes,) + similarities.shape))
        for class_id in range(n_classes):
            masks[class_id] = neighbors_labels == class_id

        knn_outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)
        knn_outputs = knn_outputs.T
        outputs = args.knn_weight * knn_outputs + (1 - args.knn_weight) * test_outputs

        predictions = np.argmax(outputs, axis=1)
        correct = (test_labels == predictions).sum()

    total_loss = criterion(torch.tensor(outputs), torch.tensor(test_labels))
    return correct, total, total_loss


def compute_accuracy_local(nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()

        if args.dataset == 'shakespeare':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

            test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)
        
        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples
    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets=None, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(global_model)
        local_model.eval()

        if args.dataset == 'shakespeare':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)

        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
     
            if args.k_neighbor:
                n_train_samples = len(train_dl_local.dataset)
                capacity = int(args.capacity_ratio * n_train_samples)
                rng = np.random.default_rng(seed=args.init_seed)
                # vec_dim = 128*65
                vec_dim = 128
                datastore = DataStore(capacity, "random", vec_dim, rng)
                
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, test_dl_local, datastore, vec_dim, args, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, train_dl_local, datastore, vec_dim, args, device=device)

            else:
                test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_per_client(hyper, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, num_class, device="cpu"):  
    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

            if args.k_neighbor:
                n_train_samples = len(train_dl_local.dataset)
                capacity = int(args.capacity_ratio * n_train_samples)
                rng = np.random.default_rng(seed=args.init_seed)
                # vec_dim = 128*65
                vec_dim = 128
                datastore = DataStore(capacity, "random", vec_dim, rng)
                
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, test_dl_local, datastore, vec_dim, args, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, train_dl_local, datastore, vec_dim, args, device=device)

            else:
                test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_percnn_client(hyper, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):

    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device))
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        elif args.noise_type == 'increasing':
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
        else:
            noise_level = 0
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_personally(personal_qkv_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):

    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):

        node_weights = personal_qkv_list[net_id]
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
           
            if args.k_neighbor:
                n_train_samples = len(train_dl_local.dataset)
                capacity = int(args.capacity_ratio * n_train_samples)
                rng = np.random.default_rng(seed=args.init_seed)
                # vec_dim = 128*65
                vec_dim = 128
                datastore = DataStore(capacity, "random", vec_dim, rng)
                
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, test_dl_local, datastore, vec_dim, args, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, train_dl_local, datastore, vec_dim, args, device=device)
            
            else:
                test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_perRod(personal_head_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, alpha_dict, device="cpu", hyper=None):
    if hyper != None:
        hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(global_model)
        if hyper!=None:
            node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
            local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        p_head = copy.deepcopy(personal_head_list[net_id])
        p_head.eval()

        sample_per_class = alpha_dict[net_id]
        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)          
            
            test_correct, test_total, test_avg_loss = compute_accuracy_fedRod(local_model, p_head, test_dl_local, sample_per_class, args, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_fedRod(local_model, p_head, train_dl_local, sample_per_class, args, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_perProto(nets, global_protos, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)          
        
        test_correct, test_total, test_avg_loss = compute_accuracy_fedproto(local_model, 
            global_protos, test_dl_local, args, device=device)
        
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_fedproto(local_model, 
            global_protos, train_dl_local, args, device=device)

            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_hphead_client(hyper, personal_qkv_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):

    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
        p_head = personal_qkv_list[net_id]
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.load_state_dict(p_head, strict=False)
        local_model.eval()

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):

        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 255)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0, apply_noise=False):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def get_divided_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test, noise_level=0, net_id=None, total=0, drop_last=False, apply_noise=False):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated
        if apply_noise:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                GaussianNoise(0., noise_level)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                GaussianNoise(0., noise_level)
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated
        if apply_noise:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                GaussianNoise(0., noise_level)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                GaussianNoise(0., noise_level)
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, dataidxs= dataidxs_test ,train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def get_spe_dataloaders(dataset, data_dir, batch_size, chunk_len, is_validation=False):

    inputs, targets = None, None
    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(os.listdir(data_dir)):
        task_data_path = os.path.join(data_dir, task_dir)

        train_iterator = get_spe_loader(dataset=dataset,
        path=os.path.join(task_data_path, f"train{EXTENSIONS[dataset]}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=True)

        val_iterator = get_spe_loader(dataset=dataset,
        path=os.path.join(task_data_path, f"train{EXTENSIONS[dataset]}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=False)

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator =get_spe_loader(dataset=dataset,
        path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[dataset]}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=False)

        if test_iterator!=None:
            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)

    original_client_num = task_id + 1

    return train_iterators, val_iterators, test_iterators, original_client_num


def get_spe_loader(dataset, path, batch_size, train, chunk_len=5, inputs=None, targets=None):

    if dataset == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=chunk_len)
    else:
        raise NotImplementedError(f"{dataset} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    drop_last = (len(dataset) > batch_size) and train

    return data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers=NUM_WORKERS)
