#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=34277 --run_split=9 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=34277 --run_split=8 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=26277 --run_split=7 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=42277 --run_split=6 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=46277 --run_split=5 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=30277 --run_split=4 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=42277 --run_split=3 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=46277 --run_split=2 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=30277 --run_split=1 --hidden=64 --num_hops=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=30277 --run_split=0 --hidden=64 --num_hops=2

#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=30277 --num_hops=2 --hidden=64
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=squirrel --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_edges=17336 --num_hops=2 --hidden=64
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_edges=16076 --num_hops=2 --hidden=64 --runs=10
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=cora --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_edges=99744 --num_hops=2 --hidden=64 --runs=10
#CPU_ONLY=1 python -m benchmark.node_classification.cheb --dataset=actor --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=0 --num_hops=3 --hidden=64

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=0 --num_edges=5278
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=1 --num_edges=3938
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=2 --num_edges=4478
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=3 --num_edges=38
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=4 --num_edges=1308
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=5 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=6 --num_edges=18
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=7 --num_edges=808
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=8 --num_edges=78
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=9 --num_edges=3558

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=0 --num_edges=7835
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=1 --num_edges=1195
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=2 --num_edges=2145
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=3 --num_edges=9045
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=4 --num_edges=1085
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=5 --num_edges=2405
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=6 --num_edges=65
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=7 --num_edges=135
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=8 --num_edges=685
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.cheb --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=3 --hidden=64 --run_split=9 --num_edges=115

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=CiteSeer --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=4 --hidden=512 --num_edges=7800 --runs=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cora --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=128 --num_edges=8000 --runs=10 --rewired

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_hops=1 --hidden=512 --run_split=0 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_hops=2 --hidden=512 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_hops=2 --hidden=512 --run_split=2 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --num_hops=2 --hidden=512 --run_split=3 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_hops=2 --hidden=512 --run_split=4 --num_edges=754
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_hops=3 --hidden=512 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_hops=3 --hidden=512 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_hops=2 --hidden=512 --run_split=6 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_hops=2 --hidden=512 --run_split=7 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_hops=2 --hidden=512 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_hops=2 --hidden=512 --run_split=9 --num_edges=252

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.2 --num_hops=2 --hidden=512 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --num_hops=2 --hidden=512 --run_split=1 --num_edges=45
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_hops=1 --hidden=512 --run_split=2 --num_edges=22
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.1 --num_hops=1 --hidden=512 --run_split=3 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.2 --num_hops=2 --hidden=512 --run_split=4 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.1 --num_hops=1 --hidden=512 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.5 --num_hops=2 --hidden=512 --run_split=6 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.5 --num_hops=2 --hidden=512 --run_split=7 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.8 --num_hops=2 --hidden=512 --run_split=8 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.8 --num_hops=2 --hidden=512 --run_split=9 --num_edges=45

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --num_hops=3 --hidden=1024 --run_split=0 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --num_hops=2 --hidden=512 --run_split=1 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --num_hops=3 --hidden=512 --run_split=2 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --num_hops=1 --hidden=512 --run_split=3 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=4 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --num_hops=1 --hidden=1024 --run_split=5 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --num_hops=1 --hidden=1024 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --num_hops=3 --hidden=512 --run_split=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=9 --num_edges=10
