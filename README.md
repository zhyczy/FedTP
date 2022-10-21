# FedTP
This is the official implementation of paper [FedTP: Federated Learning by Transformer Personalization].


Federated learning is an emerging learning paradigm where multiple clients collaboratively train a machine learning model in a privacy-preserving manner. Personalized federated learning extends this paradigm to overcome heterogeneity across clients by learning personalized models. Recent works have shown that the self-attention mechanism in Transformer models is robust to distribution shifts. As such, there have been some initial attempts to apply Transformers to federated learning. However, the impacts of federated learning algorithms on self-attention have not yet been studied.


We propose FedTP, a novel Transformer-based federated learning framework with personalized self-attention to better handle data heterogeneity among clients. FedTP learns a personalized self-attention layer for each client while the parameters of the other layers are shared among the clients. Additionally, the server learns a hypernetwork that generates projection matrices in the selfattention layers to produce client-wise queries, keys, and values. This hypernetwork is an effective way of sharing parameters across clients while maintaining the flexibility of a personalized Transformer. Notably, FedTP offers a convenient environment for performing a range of image and language tasks using the same federated network architecture –all of which benefits from Transformer personalization. Extensive experiments on standard personalized federated learning benchmarks with different non-IID data settings show that FedTP yields state-of-the-art performance.

<img src="figures/pipeline.png" width="700" height="300" /><br/>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Additionally, To run FedTP+KNN, FAISS should be installed. Instructions for the installation of FAISS can be found
[here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)


## Datasets

We provide three federated benchmark datasets spanning image classification task (CIFAR10 and CIFAR100) 
and language modeling(Shakespeare).


For CIFAR10 and CIFAR100 dataset, download and unzip data under 'data' file catalog. 
Or simply run any algorithms with CIFAR10/CIFAR100 dataset, the program will download data automatically.

Shakespeare dataset was naturally partitioned by assigning
all lines from the same characters to the same client. 
See the `README.md` files in `data/shakespeare` 
for instructions on generating data before running experiments.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| CIFAR10   |     Image classification        |      vit/cnn/cnn-b |
| CIFAR100    |     Image classification         |      vit/cnn/cnn-b  |
| Shakespeare |     Next character prediction        |      transformer/Stacked LSTM    |



## Usage
Here is one example to run our FedTP:
```
python main.py --model=vit \
    --dataset=cifar10 \
    --alg=FedTP \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=0.1 \
    --init_seed=0
```

Three demos for each dataset are provided in scripts folder.

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `cnn`, `cnn-b`, `vit`, `lstm`, `transformer`. Default = `vit`. |
| `dataset`      | Dataset to use. Options: `cifar10`, `cifar100`, `shakespeare`. Default = `cifar10`. |
| `alg` | Basic training algorithm. Basic Options: `fedavg`, `fedprox`, `FedTP`, `pFedHN`, `pfedMe`, `fedPer`, `fedBN`, `fedRod`, `fedproto`, `local_training`. Extension: `Personalized-T`, `FedTP-Per`, `FedTP-Rod`. Default = `FedTP`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `1`. |
| `n_parties` | Number of parties, default = `10`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `eval_step`    | Test interval during communication, default = `1`. |
| `test_round`    | Round beginning to test, default = `2`. |
| `partition`    | The partition way. Options: `noniid-labeldir`, `noniid-labeldir100`, `noniid-labeluni`, `iid-label100`, `homo`. Default = `noniid-labeldir` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `sample` | Ratio of parties that participate in each communication round, default = `0.1`. |
| `balanced_soft_max` | Activate this to run FedRod and FedTP-Rod. |
| `k_neighbor` | Activate this to run FedTP-KNN. |
| `init_seed` | The initial seed, default = `0`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `noise_type` | Noise type. Use `increasing` to check effect of heterogeneity in Noise-based Feature Imbalance, default = `None`. |
| `save_model` | Activate this to save model. |



## Data Partition Map
To simulate non-IID scenarios for CIFAR-10/CIFAR-100, we follow two common split designs. You can call function `get_partition_dict()` in `main.py` to access `net_dataidx_map`. `net_dataidx_map` is a dictionary. Its keys are party ID, and the value of each key is a list containing index of data assigned to this party. For our experiments, we usually set `init_seed=0`.  The default value of `noise` is 0 unless stated. We list the way to get our data partition as follow.
* **Dirichlet Partition**: `partition`=`noniid-labeldir/noniid-labeldir100`. The former is for CIFAR-10 dataset and the later is for CIFAR-100 dataset. `beta` controls degree of data heterogeneity. 
* **Pathological Partition**: `partition`=`noniid-labeluni`. For CIFAR-10 and CIFAR-100 dataset. 


Here is explanation of parameter for function `get_partition_dict()`. 

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dataset`      | Dataset to use. Options: `cifar10`, `cifar100` |
| `partition`    | Tha partition way. Options: `noniid-labeldir`, `noniid-labeldir100`, `noniid-labeluni`, `iid-label100`, `homo` |
| `n_parties` | Number of parties. |
| `init_seed` | The initial seed. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |



<!-- ## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
``` -->
