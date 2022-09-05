from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ViTHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, dim_head=64, n_hidden=1, depth=6,
                 spec_norm=False):
        super(ViTHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.to_qkv_value_list=nn.ModuleList([])
        for d in range(self.depth):
            to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim * 3)
            self.to_qkv_value_list.append(to_qkv_value)


    def finetune(self, emd):
        features = self.mlp(emd)  
        weights=OrderedDict()
        for d in range(self.depth):
            layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
            layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(self.inner_dim * 3,self.dim)
            weights["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value
        return weights


    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(-1,self.inner_dim * 3,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(self.inner_dim * 3,self.dim)
                weights["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value
        return weights


class ShakesHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, dim_head=64, n_hidden=1, depth=6,
                 spec_norm=False):
        super(ShakesHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.wqs_value_list=nn.ModuleList([])
        self.wks_value_list=nn.ModuleList([])
        self.wvs_value_list=nn.ModuleList([])

        for d in range(self.depth):
            wq_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wk_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wv_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            self.wqs_value_list.append(wq_value)
            self.wks_value_list.append(wk_value)
            self.wvs_value_list.append(wv_value)


    def finetune(self, emd):
        features = self.mlp(emd)
        weights=OrderedDict()
        for d in range(self.depth):
            layer_d_q_value_hyper = self.wqs_value_list[d]
            layer_d_q_value = layer_d_q_value_hyper(features).view(self.inner_dim ,self.dim)
            layer_d_k_value_hyper = self.wks_value_list[d]
            layer_d_k_value = layer_d_k_value_hyper(features).view(self.inner_dim ,self.dim)
            layer_d_v_value_hyper = self.wvs_value_list[d]
            layer_d_v_value = layer_d_v_value_hyper(features).view(self.inner_dim ,self.dim)
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights


    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[d]
                layer_d_q_value = layer_d_q_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_k_value_hyper = self.wks_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_v_value_hyper = self.wvs_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[d]
                layer_d_q_value = layer_d_q_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_k_value_hyper = self.wks_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_v_value_hyper = self.wvs_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(self.inner_dim ,self.dim)
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights

