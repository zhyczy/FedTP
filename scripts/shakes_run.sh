cd ..
# python main.py --model=vit \
# 	--dataset=cifar10 \
# 	--alg=FedTP \
# 	--lr=0.01 \
# 	--batch-size=64 \
# 	--epochs=1 \
# 	--n_parties=10 \
# 	--rho=0.9 \
# 	--comm_round=20 \
# 	--partition=noniid-labeldir \
# 	--beta=0.3\
# 	--device='cuda:0'\
# 	--datadir='./data/' \
# 	--logdir='./logs_emb/' \
# 	--noise=0\
# 	--init_seed=0\
# 	--sample=0.1\
# 	--eval_step=1\
# 	--calibrated=True\
# 	--k_neighbor=True \
	# --balanced_soft_max \
	# --layer_emd \

# python main.py --model=vit \
# 	--dataset=cifar100 \
# 	--alg=FedTP \
# 	--lr=0.01 \
# 	--batch-size=64 \
# 	--epochs=1 \
# 	--n_parties=100 \
# 	--rho=0.9 \
# 	--comm_round=3 \
# 	--partition=iid-label100 \
# 	--beta=0.3\
# 	--device='cuda:0'\
# 	--datadir='./data/' \
# 	--logdir='./logs_emb/' \
# 	--noise=0\
# 	--init_seed=0\
# 	--eval_step=1\
# 	--save_model \
# 	--sample=0.01

# 	--beginning_round=1\
# 	--update_round=1\
# 	--k_neighbor
	# --calibrated=True\
	# --no_mlp_head=True\
	# --hyper_hid=256\
	# --position_embedding=True\


# python main.py --model=vit \
# 	--dataset=cifar10 \
# 	--alg=pfedMe \
# 	--lr=0.01 \
# 	--batch-size=64 \
# 	--epochs=1 \
# 	--n_parties=10 \
# 	--rho=0.9 \
# 	--comm_round=2\
# 	--version=7 \
# 	--partition=noniid-labeldir\
# 	--beta=0.3\
# 	--device='cuda:0'\
# 	--datadir='./data/' \
# 	--logdir='./logs_emb/' \
# 	--noise=0\
# 	--init_seed=0\
# 	--eval_step=1\
# 	--save_model \
# 	--sample=0.1

# 	--chunk_len=10
# 	--calibrated=True

# python main.py --model=lstm\
#     --dataset=shakespeare \
#     --alg=pfedMe \
#     --lr=0.01 \
#     --batch-size=64 \
#     --epochs=1 \
#     --rho=0.9 \
#     --comm_round=1 \
#     --depth=2 \
#     --partition=noniid-labeldir \
#     --chunk_len=30\
#     --device='cuda:0'\
#     --datadir='./data/' \
#     --logdir='./logs_exp/' \
#     --init_seed=0\
#     --sample=0.01

# python main.py --model=transformer \
#   --dataset=shakespeare \
#   --alg=fedRod \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=10 \
#   --rho=0.9 \
#   --comm_round=1 \
#   --partition=noniid-labeldir \
#   --beta=0.3\
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --eval_step=1\
#   --chunk_len=10\
#   --sample=0.01
  # --use_hyperRod \


python main.py --model=lstm\
    --dataset=shakespeare \
    --alg=fedPer \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=1 \
    --rho=0.9 \
    --comm_round=1 \
    --depth=2 \
    --partition=noniid-labeldir \
    --chunk_len=30\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs_exp/' \
    --init_seed=0\
    --sample=0.01

