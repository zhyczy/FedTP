cd ..
python main.py --model=vit \
  --dataset=cifar100 \
  --alg=FedTP \
  --lr=0.01 \
  --batch-size=64 \
  --epochs=1 \
  --n_parties=10 \
  --rho=0.9 \
  --comm_round=20 \
  --partition=noniid-labeldir100 \
  --beta=0.3\
  --device='cuda:0'\
  --datadir='./data/' \
  --logdir='./logs_emb/' \
  --noise=0\
  --init_seed=0\
  --sample=0.1\
  --eval_step=1


# python main.py --model=vit \
#   --dataset=cifar100 \
#   --alg=FedTP \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=10 \
#   --rho=0.9 \
#   --comm_round=20 \
#   --partition=noniid-labeluni \
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --sample=0.1\
#   --eval_step=1


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
#   --sample=0.1