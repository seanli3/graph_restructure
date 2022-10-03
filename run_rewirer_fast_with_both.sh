CPU_ONLY=True python -m models.encoder_node_classification_fast --dataset=ogbn-arxiv --lr=0.01 --step=0.2 --sample_size=20 --eps=0.1 --with_rand_signal --with_node_feature
#CPU_ONLY=True python -m models.encoder_node_classification_fast --dataset=Penn94 --lr=0.01 --step=0.1 --sample_size=20 --eps=0.1 --with_rand_signal --with_node_feature
#CPU_ONLY=True python -m models.encoder_node_classification_fast --dataset=Wisconsin --lr=0.01 --step=0.1 --sample_size=20 --eps=0.1 --with_rand_signal --with_node_feature --split=0
#CPU_ONLY=True python -m models.encoder_node_classification_fast --dataset=ogbn-arxiv --lr=0.01 --step=0.1 --sample_size=20 --eps=0.1 --with_rand_signal --with_node_feature
