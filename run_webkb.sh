#!/bin/sh

CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=0
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=1
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=2
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=3
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=4
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=5
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=6
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=7
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=8
CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m models.encoder_node_classification --dataset=texas --step=0.1 --lr=0.01 --mode=supervised --split=9
