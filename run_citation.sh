#!/bin/sh


#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=cora --step=0.1 --lr=0.01 --mode=supervised
#CPU_ONLY=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=citeseer --step=0.1 --lr=0.01 --mode=supervised
CPU_ONLY=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=pubmed --step=0.1 --lr=0.01 --mode=supervised
