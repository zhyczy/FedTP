cd ..
python main.py --model=transformer \
  --dataset=shakespeare \
  --alg=FedTP \
  --lr=0.01 \
  --batch-size=64 \
  --epochs=1 \
  --n_parties=10 \
  --rho=0.9 \
  --comm_round=2 \
  --device='cuda:0'\
  --datadir='./data/' \
  --logdir='./logs_emb/' \
  --noise=0\
  --depth=2\
  --init_seed=0\
  --eval_step=1\
  --chunk_len=10\
  --sample=0.1


# python main.py --model=lstm\
#     --dataset=shakespeare \
#     --alg=pfedMe \
#     --lr=0.01 \
#     --batch-size=64 \
#     --epochs=1 \
#     --rho=0.9 \
#     --comm_round=2 \
#     --depth=2 \
#     --chunk_len=30\
#     --device='cuda:0'\
#     --datadir='./data/' \
#     --logdir='./logs_exp/' \
#     --init_seed=0\
#     --sample=0.005