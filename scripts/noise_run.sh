cd ..
# python main.py --model=vit \
# 	--dataset=cifar10 \
# 	--alg=protoVit \
# 	--lr=0.01 \
# 	--batch-size=64 \
# 	--epochs=1 \
# 	--n_parties=10 \
# 	--rho=0.9 \
# 	--comm_round=10 \
# 	--partition=noniid-labeldir \
# 	--beta=0.3\
# 	--device='cuda:0'\
# 	--datadir='./data/' \
# 	--logdir='./logs_emb/' \
# 	--noise=0\
# 	--init_seed=0\
# 	--sample=0.1\
# 	--beginning_round=1\
# 	--update_round=5\
# 	--eval_step=5\
	# --save_model=True
	# --k_neighbor=True

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
# 	--k_neighbor=True \
	# --balanced_soft_max 

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
	# --hyper_hid=256


# python main.py --model=vit \
# 	--dataset=cifar10 \
# 	--alg=FedTP-Per \
# 	--lr=0.01 \
# 	--batch-size=64 \
# 	--epochs=1 \
# 	--rho=0.9 \
# 	--depth=6 \
# 	--comm_round=1 \
# 	--partition=noniid-labeldir \
# 	--device='cuda:0'\
# 	--datadir='./data/' \
# 	--logdir='./logs_exp/' \
# 	--chunk_len=5\
# 	--init_seed=0\
# 	--eval_step=1
# 	--sample=0.01

	
# python main.py --model=cnn \
# 	--dataset=cifar10 \
# 	--alg=pFedHN \
# 	--lr=0.01 \
# 	--batch-size=64 \
# 	--epochs=1 \
# 	--n_parties=10 \
# 	--rho=0.9 \
# 	--comm_round=2000 \
# 	--partition=noniid-labeldir \
# 	--beta=0.3\
# 	--device='cuda:0'\
# 	--datadir='./data/' \
# 	--logdir='./logs_exp/' \
# 	--noise=0\
# 	--init_seed=0\
# 	--sample=0.1\
	# --hyper_hid=256


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


# python main.py --model=cnn \
#   --dataset=cifar10 \
#   --alg=fedRod \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=50 \
#   --rho=0.9 \
#   --comm_round=1500 \
#   --partition=noniid-labeldir \
#   --beta=0.3\
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --eval_step=10\
#   --chunk_len=10\
#   --sample=0.1 \
#   --balanced_soft_max


# python main.py --model=vit \
#   --dataset=cifar10 \
#   --alg=local_training \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=50 \
#   --rho=0.9 \
#   --comm_round=200 \
#   --partition=noniid-labeldir \
#   --beta=0.3\
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --eval_step=1\
#   --chunk_len=10\
#   --definite_selection \
#   --save_model \
#   --sample=0.1

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

# python main.py --model=vit \
#   --dataset=cifar10 \
#   --alg=FedTP-Rod \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=10 \
#   --rho=0.9 \
#   --comm_round=3 \
#   --partition=noniid-labeldir \
#   --beta=0.3\
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --eval_step=1\
#   --sample=0.1\
#   --balanced_soft_max

# python main.py --model=vit \
#   --dataset=cifar10 \
#   --alg=fedproto \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=2\
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
#   --sample=0.1

# python main.py --model=lstm\
#     --dataset=shakespeare \
#     --alg=fedPer \
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

# python main.py --model=cnn-b \
#   --dataset=cifar100 \
#   --alg=fedBN \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=50 \
#   --rho=0.9 \
#   --comm_round=10 \
#   --partition=noniid-labeldir100 \
#   --beta=0.3\
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --eval_step=1\
#   --sample=0.1\


python main.py --model=cnn-b \
  --dataset=cifar10 \
  --noise=20\
  --alg=fedBN \
  --lr=0.01 \
  --batch-size=64 \
  --epochs=1 \
  --n_parties=10 \
  --rho=0.9 \
  --comm_round=500 \
  --test_round=2 \
  --eval_step=1 \
  --partition=homo \
  --device='cuda:0'\
  --datadir='./data/' \
  --logdir='./logs_emb/' \
  --noise_type='increasing' \
  --init_seed=0\
  --sample=0.5

  # --k_neighbor=True \
