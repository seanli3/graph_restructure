#CPU_ONLY=True python -m models.encoder_node_classification_fast --dataset=ogbn-arxiv --lr=0.01 --step=0.2 --mode=supervised --sample_size=10 --eps=0.1 --with_rand_signal --with_node_feature
#CPU_ONLY=True python -m models.encoder_node_classification_fast --dataset=Penn94 --lr=0.01 --step=0.1 --mode=supervised --sample_size=10 -jeps=0.1 --with_rand_signal --with_node_feature
CUDA_DEVICE=1 python -m models.encoder_node_classification_fast --dataset=ogbn-arxiv --lr=0.01 --step=0.1 --mode=supervised --sample_size=20 --eps=0.1 --with_rand_signal --with_node_feature
