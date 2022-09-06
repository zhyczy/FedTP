import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from collections import OrderedDict, defaultdict
from pathlib import Path
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
from torch.utils.tensorboard import SummaryWriter

from models.vit import ViT
from models.Hypernetworks import ViTHyper, ShakesHyper
from models.cnn import CNNHyper, CNNTarget, CNN_B
from models.language_transformer import Transformer
from models.lstm import NextCharacterLSTM
from utils import *
from methods.method import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='FedTP',  help='communication strategy: fedavg/FedTP/Personalized-T/pFedHN/pfedMe/fedprox/fedPer/fedBN/fedRod/fedproto/local_training/FedTP-Per/FedTP-Rod')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='None', help='Noise type: None/increasng/space')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--train_acc_pre', action='store_true')
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--test_round', type=int, default=2)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--comment", default="_")
    parser.add_argument("--definite_selection", action='store_true')
    parser.add_argument("--show_all_accuracy", action='store_true')
    parser.add_argument("--version", type=int, default=1)

    """
    Used for FedTP
    """
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hyper_hid', type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--balanced_soft_max", action='store_true')
    parser.add_argument("--client_embed_size", type=int, default=128)
    parser.add_argument('--log_flag', default=True)
    parser.add_argument('--k_neighbor', action='store_true')

    parser.add_argument('--capacity_ratio', type=float, default=1.0)
    parser.add_argument('--k_value', default=10)
    parser.add_argument('--knn_weight', type=float, default=0.6)

    """
    Used for shakespeare
    """
    parser.add_argument('--chunk_len', type=int, default=5)

    """
    Used for pfedMe
    """
    parser.add_argument('--pfedMe_k', type=int, default=5)
    parser.add_argument('--pfedMe_lambda', type=float, default=15)
    parser.add_argument('--pfedMe_beta', type=float, default=1)
    parser.add_argument('--pfedMe_mu', type=float, default=0)

    """
    Used for fedproto
    standard deviation: 2; rounds: 110; weight of proto loss: 0.1 
    local_bs 32

    """
    parser.add_argument('--fedproto_lambda', default=0.1)

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):

        if args.model == "vit":
            if args.dataset == "cifar10":
                net = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 128, depth = args.depth, heads = 8, mlp_dim = 256,
                  dropout = 0.1, emb_dropout = 0.1)      
            elif args.dataset == "cifar100":
                net = ViT(image_size=32, patch_size=4, num_classes=100, dim=128, depth = args.depth, heads=8, mlp_dim=256,
                          dropout=0.1, emb_dropout=0.1)

        elif args.model == "cnn":
            if args.dataset == "cifar10":
                net = CNNTarget(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNNTarget(n_kernels=16, out_dim=100)

        elif args.model == "cnn-b":
            if args.dataset == "cifar10":
                net = CNN_B(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNN_B(n_kernels=16, out_dim=100)

        elif args.model == "lstm":
            net = NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"])

        elif args.model == "transformer":
            net = Transformer(n_src_vocab=len(string.printable), 
            n_trg_vocab=len(string.printable),
            d_k=64, d_v=64, d_model=128,
            d_word_vec=128, d_inner=256,
            n_layers=args.depth, n_head=8, dropout=0.1)

        else:
            raise NotImplementedError("not supported yet")

        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_hyper(args, sam_node=None):
    # embed_dim = int(1 + args.n_parties / 4)  
    embed_dim = args.client_embed_size
    batch_node = int(args.n_parties * args.sample)
    if args.model == "vit":   
        if args.dataset == "cifar10":
            hnet = ViTHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128, 
                      heads = 8, dim_head = 64, n_hidden = args.n_hidden, depth=args.depth, client_sample=batch_node)
        
        elif args.dataset == "cifar100":
            hnet = ViTHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128,
                        heads=8, dim_head=64, n_hidden = args.n_hidden, depth=args.depth, client_sample=batch_node)
    
    elif args.model == "cnn":
        if args.dataset == "cifar10":
            hnet = CNNHyper(args.n_parties, embed_dim, hidden_dim=args.hyper_hid, n_hidden=args.n_hidden, n_kernels=16)
        
        elif args.dataset == "cifar100":
            hnet = CNNHyper(args.n_parties, embed_dim, hidden_dim=args.hyper_hid,
                            n_hidden=args.n_hidden, n_kernels=16, out_dim=100)

    elif args.model == "transformer":
        if args.dataset != "shakespeare":
            raise NotImplementedError("ShakesHyper only supports shakespeare dataset.")
        
        hnet = ShakesHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128, 
        heads = 8, dim_head = 64, n_hidden = args.n_hidden, 
        depth=args.depth, client_sample=sam_node)

    return hnet


def init_personalized_parameters(args, client_number=None):
    personalized_pred_list = []
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100
    elif args.dataset == "shakespeare":
        class_num = 100

    if args.alg == 'Personalized-T':
        if args.model == 'vit':
            for nndx in range(args.n_parties):
                kqv_dict = OrderedDict()
                for ll in range(args.depth):
                    kqv_dict["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"]=None
                personalized_pred_list.append(kqv_dict)
        elif args.model == 'transformer':
            for nndx in range(client_number):
                kqv_dict = OrderedDict()
                for ll in range(args.depth):
                    kqv_dict["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"]=None
                    kqv_dict["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"]=None
                    kqv_dict["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"]=None
                personalized_pred_list.append(kqv_dict)

    elif args.alg == 'fedRod':
        if args.model == "cnn":
            dim = 84
            for nndx in range(args.n_parties):
                p_class = nn.Linear(dim, class_num).to(args.device)
                personalized_pred_list.append(p_class)
        elif args.model in ["vit", "transformer"]:
            dim = 128
            for nndx in range(args.n_parties):
                p_class = nn.Linear(dim, class_num).to(args.device)
                personalized_pred_list.append(p_class)

    elif args.alg == 'fedPer':
        if args.model == 'cnn':
            dim = 84
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                para_dict["fc3.weight"] = None
                para_dict["fc3.bias"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'vit':
            dim = 128
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                para_dict["mlp_head.1.weight"] = None
                para_dict["mlp_head.1.bias"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'transformer':
            dim = 128
            for nndx in range(client_number):
                para_dict = OrderedDict()
                para_dict["trg_word_prj.weight"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'lstm':
            dim = 256
            for nndx in range(client_number):
                para_dict = OrderedDict()
                para_dict["decoder.weight"] = None
                para_dict["decoder.bias"] = None
                personalized_pred_list.append(para_dict)

    elif args.alg == 'fedBN':
        for nndx in range(args.n_parties):
            bn_dict = OrderedDict()
            for ll in range(4):
                bn_dict["bn"+str(ll+1)+".weight"] = None
                bn_dict["bn"+str(ll+1)+".bias"] = None
            personalized_pred_list.append(bn_dict)

    elif args.alg == 'FedTP-Rod':
        dim = 128
        for nndx in range(args.n_parties):
            p_class = nn.Linear(dim, class_num).to(args.device)
            personalized_pred_list.append(p_class)

    elif args.alg == 'FedTP-Per':
        dim = 128
        for nndx in range(args.n_parties):
            para_dict = OrderedDict()
            para_dict["mlp_head.1.weight"] = None
            para_dict["mlp_head.1.bias"] = None
            personalized_pred_list.append(para_dict)

    return personalized_pred_list


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map_train, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map_train


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    logging.info("Dataset: %s" % args.dataset)
    logging.info("Backbone: %s" % args.model)
    logging.info("Method: %s" % args.alg)
    logging.info("Partition: %s" % args.partition)
    logging.info("Beta: %f" % args.beta)
    logging.info("Sample rate: %f" % args.sample)
    logging.info("Print Accuracy on training set: %s" % args.train_acc_pre)
    logging.info("Save model: %s" % args.save_model)
    logging.info("Total running round: %s" % args.comm_round)
    logging.info("Test round fequency: %d" % args.eval_step)
    logging.info("Noise Type: %s" %args.noise_type)
    logging.info("Show every client's accuracy: %s" %args.show_all_accuracy)
    if args.noise_type != 'None':
        if args.partition != 'homo':
            raise NotImplementedError("Noise based feature skew only supports iid partition")
        logging.info("Max Noise: %d" %args.noise)
    if args.model in ["vit", "transformer"]:
        logging.info("Transformer depth: %d" % args.depth)
        if args.alg in ["FedTP", "FedTP-Rod"]:
            logging.info("Hyper hidden dimension: %d" % args.hyper_hid)
            logging.info("Client embedding size: %d" %args.client_embed_size)
            logging.info("Use balance soft-max: %s" %args.balanced_soft_max)
    if args.dataset == "shakespeare":
        if args.model not in ['lstm', 'transformer']:
            raise NotImplementedError("Serial data needs lstm, transformer as backbone.")
    if args.alg == "fedprox":
        logging.info("mu value: %f" %args.mu)
    if args.test_round<=1:
        raise NotImplementedError("test round should be larger than 1")

    save_path = args.alg+"-"+args.model+"-"+str(args.n_parties)+"-"+args.dataset+"-"+args.partition+args.comment
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    if args.k_neighbor:
        logging.info("Use memory: %s" % args.k_neighbor)

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path= args.alg + " " + args.model + " " + str(args.version) + '-experiment_arguments-%s.json ' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = args.model + " " + str(args.version) + '-experiment_log-%s ' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    eval_step = args.eval_step
    acc_all = []

    logger.info("Partitioning data")

    if args.dataset != 'shakespeare':
        # if args.alg=="local_training":
        #     args.test_round=41
        logging.info("Test beginning round: %d" %args.test_round)
        logging.info("Client Number: %d" % args.n_parties)
        X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
            args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir)

        n_classes = len(np.unique(y_train))
        train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        logger.info("len train_dl_global: %d"  %len(train_ds_global))
        data_size = len(test_ds_global)

    else:
        if args.model not in ["lstm", "transformer"]:
            raise NotImplementedError("shakespeare supports lstm or transformer")
        data_dir = os.path.join("data", "shakespeare", "all_data", "train")
        train_dl_global, val_dl_global, test_dl_global, original_c_num = get_spe_dataloaders(args.dataset, data_dir, args.batch_size, args.chunk_len)
        args.n_parties = len(train_dl_global)
        
        # if args.alg=="local_training":
        #     args.test_round=21
        logging.info("Test beginning round: %d" %args.test_round)
        logger.info("Drop Client Number: %d" %(original_c_num-args.n_parties))
        logger.info("Client Number: %d" % args.n_parties)
        logger.info("Chunk_len: %d" %args.chunk_len)

    results_dict = defaultdict(list)
    eval_step = args.eval_step
    best_step = 0
    best_accuracy = -1
    test_round = args.test_round

    if args.alg == 'fedavg':
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                    global_model, args, train_dl_global, test_dl_global, nets, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                    global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'FedTP':
        if args.model not in ["vit", "transformer"]:
            raise NotImplementedError("FedTP only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        if args.dataset == "shakespeare":
            sam_node = int(args.n_parties * args.sample)
            hnet = init_hyper(args, sam_node).to(device)
        else:
            hnet = init_hyper(args).to(device)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']


        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = hnet(torch.tensor([selected], dtype=torch.long).to(device),False)

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        node_weights = weights[ix]
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
                        nets[idx].load_state_dict(node_weights,strict=False)
            else:
                for ix in range(len(selected)):
                    node_weights = weights[ix]
                    idx = selected[ix]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights,strict=False)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()
                net_para = nets[selected[idx]].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
                )

                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            global_model.load_state_dict(global_para)
            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if args.dataset == "cifar10":
                num_class = 10
            elif args.dataset == "cifar100":
                num_class = 100

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(
                    hnet, nets, global_model, args, train_dl_global, test_dl_global, 0, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(
                    hnet, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, num_class, device=device)
                
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_all_acc'] = test_all_acc
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'pFedHN':
        if args.model != "cnn":
            raise NotImplementedError("pFedHN only supports cnn backbone")
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        hnet = init_hyper(args).to(device)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = {}

            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        idx = selected[ix]
                        node_weights = hnet(torch.tensor([idx], dtype=torch.long).to(device))
                        weights[ix] = node_weights
                        nets[idx].load_state_dict(node_weights)
            else:
                for ix in range(len(selected)):
                    idx = selected[ix]
                    node_weights = hnet(torch.tensor([idx], dtype=torch.long).to(device))
                    weights[ix] = node_weights
                    nets[idx].load_state_dict(node_weights)

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
                )

                if idx == 0:
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_percnn_client(
                    hnet, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device=device)
            
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'cnn_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            if args.dataset == 'shakespeare':
                local_train_net_fedprox(nets, selected, global_model, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                    global_model, args, train_dl_global, test_dl_global, nets, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                    global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, device=device) 

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'pfedMe':
        if args.model not in ['lstm', 'transformer', 'vit', 'cnn']:
            raise NotImplementedError("pfedMe only supports lstm, cnncifar backbone")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            if args.dataset == 'shakespeare':
                update_dict = local_train_net_pfedMe(nets, selected, global_model, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                update_dict = local_train_net_pfedMe(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            # prev_global = copy.deepcopy(list(global_model.parameters()))

            for param in global_model.parameters():
                param.data.zero_()

            for idx in range(len(selected)):
                net_para = update_dict[selected[idx]]
                if idx == 0:
                    for param, new_param in zip(global_model.parameters(), net_para):
                        param.data = new_param.data.clone() * fed_avg_freqs[idx]
                else:
                    for param, new_param in zip(global_model.parameters(), net_para):
                        param.data += new_param.data.clone() * fed_avg_freqs[idx]
            # for pre_param, param in zip(prev_global, global_model.parameters()):
            #     param.data = (1 - args.pfedMe_beta)*pre_param.data + args.pfedMe_beta*param.data

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                    nets, args, train_dl_global, test_dl_global, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                    nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
         
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            # chose_arr = [3,4,5]
            chose_arr = np.arange(args.n_parties)
            for node_idx in chose_arr:
                outfile_vit = os.path.join(save_path, 'Vit_'+str(node_idx)+'_1500.tar')
                torch.save({'epoch':args.comm_round+1, 'state':nets[node_idx].state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)  

    elif args.alg == 'fedPer':
        if args.model not in ['cnn', 'vit', 'transformer', 'lstm']:
            raise NotImplementedError("fedPer uses cnn as backbone")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        logger.info("Initializing Personalized Classification head")
        if args.dataset == "shakespeare":
            client_number = int(args.n_parties)
            personalized_pred_list = init_personalized_parameters(args, client_number)
        else:
            personalized_pred_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    node_weights = personalized_pred_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    if args.dataset=="shakespeare":
                        if args.model == 'lstm':
                            personalized_pred_list[iidx]["decoder.weight"] = copy.deepcopy(final_state["decoder.weight"])
                            personalized_pred_list[iidx]["decoder.bias"] = copy.deepcopy(final_state["decoder.bias"])
                        elif args.model == 'transformer':
                            personalized_pred_list[iidx]["trg_word_prj.weight"] = copy.deepcopy(final_state["trg_word_prj.weight"])
                    else:
                        if args.model == 'cnn':
                            personalized_pred_list[iidx]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                            personalized_pred_list[iidx]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])
                        elif args.model == 'vit':
                            personalized_pred_list[iidx]["mlp_head.1.weight"] = copy.deepcopy(final_state["mlp_head.1.weight"])
                            personalized_pred_list[iidx]["mlp_head.1.bias"] = copy.deepcopy(final_state["mlp_head.1.bias"])
           
            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    if args.dataset=="shakespeare":
                        if args.model == 'lstm':
                            personalized_pred_list[selected[idx]]["decoder.weight"] = copy.deepcopy(final_state["decoder.weight"])
                            personalized_pred_list[selected[idx]]["decoder.bias"] = copy.deepcopy(final_state["decoder.bias"])
                        elif args.model == 'transformer':
                            personalized_pred_list[selected[idx]]["trg_word_prj.weight"] = copy.deepcopy(final_state["trg_word_prj.weight"])
                    else:
                        if args.model == 'cnn':
                            personalized_pred_list[selected[idx]]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                            personalized_pred_list[selected[idx]]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])
                        elif args.model == 'vit':
                            personalized_pred_list[selected[idx]]["mlp_head.1.weight"] = copy.deepcopy(final_state["mlp_head.1.weight"])
                            personalized_pred_list[selected[idx]]["mlp_head.1.bias"] = copy.deepcopy(final_state["mlp_head.1.bias"])
                
                net_para = nets[selected[idx]].state_dict()               
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == "shakespeare":
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_pred_list, global_model, args, train_dl_global, test_dl_global, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedRod':
        if args.dataset == 'shakespeare':
            raise NotImplementedError("fedRod does not run on shakespeare.")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.dataset == "cifar10":
            class_num = 10
        elif args.dataset == "cifar100":
            class_num = 100
        alpha_dict = {}
        for kd, kv in traindata_cls_counts.items():
            input_embedding = torch.zeros(class_num).to(device)
            sum_v = sum(kv.values())
            for indx, indv in kv.items():
                input_embedding[indx] = indv/sum_v
            alpha_dict[kd] = input_embedding
        
        logger.info("Initializing Personalized Classification head")
        assert args.balanced_soft_max
        personalized_pred_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)

            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            update_dict = local_train_net_fedRod(nets, selected, personalized_pred_list, args, 
            net_dataidx_map_train, net_dataidx_map_test, logger, alpha=alpha_dict, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):             
                net_para = nets[selected[idx]].state_dict()  
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_perRod(
                personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, alpha_dict, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        accessories = args.alg + "-linear-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedproto':
        if args.dataset == 'shakespeare':
            raise NotImplementedError("fedproto does not run on shakespeare.")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        logger.info("Initializing prototypes")
        global_protos = None

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)

            g_protos = local_train_net_fedproto(nets, arr, args, net_dataidx_map_train, net_dataidx_map_test, 
            global_protos, logger, device=device)

            # update global model
            global_protos = copy.deepcopy(g_protos)
            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_perProto(
                nets, global_protos, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)       
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            chose_arr = [4,5,7,11,49]
            for node_idx in chose_arr:
                outfile_vit = os.path.join(save_path, 'Vit_'+str(node_idx)+'_1500.tar')
                torch.save({'epoch':args.comm_round+1, 'state':nets[node_idx].state_dict()}, outfile_vit)
            
            stored_protos = {}
            for kkk, vvv in global_protos.items():
                stored_protos[kkk] = vvv.cpu().numpy().tolist()
            proto_opt = "global_protos.json"
            with open(str(save_path / proto_opt), "w") as file:
                json.dump(stored_protos, file, indent=4)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedBN':
        if args.dataset == 'shakespeare':
            raise NotImplementedError("fedBN does not run on shakespeare.")
        if args.model != 'cnn-b':
            raise NotImplementedError("fedBN uses cnn with BN.")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        logger.info("Initializing Personalized BN layer")
        personalized_bn_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    node_weights = personalized_bn_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    for ll in range(4):
                        personalized_bn_list[iidx]["bn"+str(ll+1)+".weight"] = copy.deepcopy(final_state["bn"+str(ll+1)+".weight"])
                        personalized_bn_list[iidx]["bn"+str(ll+1)+".bias"] = copy.deepcopy(final_state["bn"+str(ll+1)+".bias"])
           
            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    for ll in range(4):
                        personalized_bn_list[selected[idx]]["bn"+str(ll+1)+".weight"] = copy.deepcopy(final_state["bn"+str(ll+1)+".weight"])
                        personalized_bn_list[selected[idx]]["bn"+str(ll+1)+".bias"] = copy.deepcopy(final_state["bn"+str(ll+1)+".bias"])
                
                net_para = nets[selected[idx]].state_dict()               
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0: 
                
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                personalized_bn_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_bn_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_bn_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        
        if args.definite_selection:
            arr = np.asarray([0,4,5,7,11,26,49])
            logging.info("Stored Clients: %s" %str(arr))
            args.epochs = 1500
            test_round = 1
        else:
            arr = np.arange(args.n_parties)

        for round in range(test_round):
            logger.info("in comm round: %d" %round)

            # if args.definite_selection == False:
            #     if round == 0:
            #         if args.dataset!="shakespeare":
            #             args.epochs=1300
            #         else:
            #             args.epochs=200
            #     else:
            #         args.epochs=5 
            # logger.info("epochs: %d" %args.epochs)
            # logger.info("test_round: %d" %test_round)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, arr, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, arr, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)
            
            if args.dataset == 'shakespeare':
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                nets, args, train_dl_global, test_dl_global, device=device)
            else:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)


            if args.log_flag:
                logger.info('>> Average Train accuracy: %f' % train_acc)
                logger.info('>> Average Test accuracy: %f' % test_acc)
                logger.info('>> Test avg loss: %f' %test_avg_loss)

            results_dict['train_avg_loss'].append(train_avg_loss)
            results_dict['train_avg_acc'].append(train_acc)
            results_dict['test_avg_loss'].append(test_avg_loss)
            results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            for node_idx in arr:
                outfile_vit = os.path.join(save_path, 'Vit_'+str(node_idx)+'_1500.tar')
                torch.save({'epoch':args.comm_round+1, 'state':nets[node_idx].state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'Personalized-T':
        if args.model not in  ["vit", "transformer"]:
            raise NotImplementedError("Personalized-T only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        logger.info("Initializing Personalized QKV heads")
        if args.dataset == "shakespeare":
            client_number = int(args.n_parties)
            personalized_kqv_list = init_personalized_parameters(args, client_number)
        else:
            personalized_kqv_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
            else:
                for ix in range(len(selected)):
                    idx = selected[ix]
                    node_weights = personalized_kqv_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights,strict=False)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = nets[iidx].state_dict()
                    for ll in range(args.depth):
                        if args.dataset=="shakespeare":
                            personalized_kqv_list[iidx]["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"])
                            personalized_kqv_list[iidx]["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"])
                            personalized_kqv_list[iidx]["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"])
                        else:
                            personalized_kqv_list[iidx]["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"] = copy.deepcopy(final_state["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"])

            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    for ll in range(args.depth):
                        if args.dataset=="shakespeare":
                            personalized_kqv_list[selected[idx]]["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"])
                            personalized_kqv_list[selected[idx]]["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"])
                            personalized_kqv_list[selected[idx]]["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"])
                        else:
                            personalized_kqv_list[selected[idx]]["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"] = copy.deepcopy(final_state["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"])
                net_para = nets[selected[idx]].state_dict()

                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == "shakespeare":
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_kqv_list, global_model, args, train_dl_global, test_dl_global, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_kqv_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + args.dataset + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)
            for ele in range(len(personalized_kqv_list)):
                p_qkv = os.path.join(save_path, 'QKV_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_kqv_list[ele]}, p_qkv)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'FedTP-Per':
        if args.model not in ["vit", "transformer"]:
            raise NotImplementedError("FedTP only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        hnet = init_hyper(args).to(device)

        logger.info("Initializing Personalized Classification head")
        personalized_pred_list = init_personalized_parameters(args)
        
        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = hnet(torch.tensor([selected], dtype=torch.long).to(device),False)

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        node_weights = weights[ix]
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
                        nets[idx].load_state_dict(node_weights, strict=False)
            else:
                for ix in range(len(selected)):
                    node_weights = weights[ix]
                    idx = selected[ix]
                    p_head = personalized_pred_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)
                    nets[idx].load_state_dict(p_head, strict=False)

            update_dict = local_train_net_per(nets, selected, args, 
            net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    personalized_pred_list[iidx]["mlp_head.1.weight"] = copy.deepcopy(final_state["mlp_head.1.weight"])
                    personalized_pred_list[iidx]["mlp_head.1.bias"] = copy.deepcopy(final_state["mlp_head.1.bias"])
      
            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()
                if round != 0:
                    personalized_pred_list[selected[idx]]["mlp_head.1.weight"] = copy.deepcopy(final_state["mlp_head.1.weight"])
                    personalized_pred_list[selected[idx]]["mlp_head.1.bias"] = copy.deepcopy(final_state["mlp_head.1.bias"])
                
                net_para = nets[selected[idx]].state_dict() 
                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True)

                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            global_model.load_state_dict(global_para)
            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_hphead_client(
                hnet, personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'FedTP-Rod':
        if args.model not in ["vit", "transformer"]:
            raise NotImplementedError("FedTP only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        hnet = init_hyper(args).to(device)

        logger.info("Initializing Personalized Classification head")
        personalized_pred_list = init_personalized_parameters(args)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']

        if args.dataset == "cifar10":
            class_num = 10
        elif args.dataset == "cifar100":
            class_num = 100
        alpha_dict = {}
        for kd, kv in traindata_cls_counts.items():
            input_embedding = torch.zeros(class_num).to(device)
            sum_v = sum(kv.values())
            for indx, indv in kv.items():
                input_embedding[indx] = indv/sum_v
            alpha_dict[kd] = input_embedding

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = hnet(torch.tensor([selected], dtype=torch.long).to(device),False)

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        node_weights = weights[ix]
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
                        nets[idx].load_state_dict(node_weights,strict=False)
            else:
                for ix in range(len(selected)):
                    node_weights = weights[ix]
                    idx = selected[ix]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights,strict=False)

            update_dict = local_train_net_fedRod(nets, selected, personalized_pred_list, args, 
            net_dataidx_map_train, net_dataidx_map_test, logger, alpha=alpha_dict, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()
                net_para = nets[selected[idx]].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
                )

                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            global_model.load_state_dict(global_para)
            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if args.dataset == "cifar10":
                num_class = 10
            elif args.dataset == "cifar100":
                num_class = 100

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_perRod(
                personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, alpha_dict, device=device, hyper=hnet)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    acc_all  = np.asarray(results_dict['test_avg_acc'])
    logger.info("Accuracy Record: ")
    logger.info(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    logger.info('Test Acc = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
    if args.show_all_accuracy:
        logger.info("Accuracy in each client: ")
        logger.info(results_dict['test_all_acc'])