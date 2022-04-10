#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=34277 --run_split=9
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=34277 --run_split=8
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=26277 --run_split=7
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=42277 --run_split=6
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=46277 --run_split=5
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=30277 --run_split=4
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=42277 --run_split=3
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=46277 --run_split=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=30277 --run_split=1
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --hidden=128 --model_indices 0 --num_edges=30277 --run_split=0

#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_edges=17336 --hidden=128 --heads=8
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_edges=16076 --hidden=256 --heads=8 --runs=10
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cora --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_edges=99744 --hidden=128 --runs=1 --heads=8 --runs=10
#CPU_ONLY=1 python -m benchmark.node_classification.gat --dataset=actor --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=0 --hidden=32 --heads=8


#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=5278 --hidden=512 --run_split=0 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=3938 --hidden=512 --run_split=1 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=4478 --hidden=512 --run_split=2 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=38 --hidden=512 --run_split=3 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=1308 --hidden=512 --run_split=4 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=28 --hidden=512 --run_split=5 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=18 --hidden=512 --run_split=6 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=808 --hidden=512 --run_split=7 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=78 --hidden=512 --run_split=8 --heads=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=3558 --hidden=512 --run_split=9 --heads=8

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=512 --run_split=0 --heads=8 --num_edges=7835
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --hidden=512 --run_split=1 --heads=8 --num_edges=1195
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=512 --run_split=2 --heads=8 --num_edges=2145
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=3 --heads=8 --num_edges=9045
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=4 --heads=8 --num_edges=1085
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=5 --heads=8 --num_edges=2405
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=6 --heads=8 --num_edges=65
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=7 --heads=8 --num_edges=135
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=8 --heads=8 --num_edges=685
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=9 --heads=8 --num_edges=115

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=512 --run_split=0 --heads=8 --num_edges=168
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=512 --run_split=1 --heads=8 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=512 --run_split=2 --heads=8 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=3 --heads=8 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=4 --heads=8 --num_edges=138
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=512 --run_split=5 --heads=8 --num_edges=58
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=6 --heads=8 --num_edges=48
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=512 --run_split=7 --heads=8 --num_edges=388
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=8 --heads=8 --num_edges=38
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gat --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=9 --heads=8 --num_edges=28

#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=CiteSeer --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=64 --num_edges=8000 --runs=10 --rewired --heads=8
# CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cora --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=128 --num_edges=8060 --runs=10 --rewired


#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --heads=8 --hidden=512 --run_split=0 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --heads=8 --hidden=128 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=128 --run_split=2 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --heads=8 --hidden=512 --run_split=3 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --heads=8 --hidden=128 --run_split=4 --num_edges=754
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=256 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=64 --run_split=6 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=64 --run_split=7 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --heads=8 --hidden=64 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --heads=8 --hidden=256 --run_split=9 --num_edges=252

#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --heads=10 --hidden=512 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --heads=10 --hidden=1024 --run_split=1 --num_edges=1789
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.5 --heads=10 --hidden=512 --run_split=2 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=10 --hidden=256 --run_split=3 --num_edges=22
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=12 --hidden=512 --run_split=4 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=10 --hidden=512 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --heads=10 --hidden=512 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --heads=10 --hidden=512 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=10 --hidden=512 --run_split=8 --num_edges=1578
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --heads=10 --hidden=512 --run_split=9 --num_edges=45

#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --heads=12 --hidden=2048 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --heads=8 --hidden=512 --run_split=1 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --heads=8 --hidden=512 --run_split=2 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --heads=8 --hidden=512 --run_split=3 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --heads=8 --hidden=1024 --run_split=4 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --heads=8 --hidden=2048 --run_split=5 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --heads=10 --hidden=2048 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --heads=8 --hidden=1024 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --heads=8 --hidden=512 --run_split=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --heads=8 --hidden=1024 --run_split=9 --num_edges=10
