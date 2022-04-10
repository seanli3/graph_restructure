#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=34277 --run_split=9
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=34277 --run_split=8
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=26277 --run_split=7
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=42277 --run_split=6
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=46277 --run_split=5
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=30277 --run_split=4
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=42277 --run_split=3
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=46277 --run_split=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=30277 --run_split=1
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=30277 --run_split=0

#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=30277


#CPU_ONLY=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_edges=90000 --hidden=512 --runs=10
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --num_edges=90000 --hidden=512 --runs=10

#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=squirrel --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=17336 --hidden=128

#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=16076 --hidden=256 --runs=10

#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cora --rewired --rewirer_step=0.1 --dropout=0.95 --model_indices 0 --num_edges=99744 --hidden=128 --runs=1

#CPU_ONLY=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cora --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=1--hidden=256 --runs=10

#CPU_ONLY=1 python -m benchmark.node_classification.gcn --dataset=actor --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=0 --hidden=256
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --num_edges=1790 --hidden=128 --run_split=0


#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --hidden=512 --run_split=0 --num_edges=168
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=512 --run_split=1 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=512 --run_split=2 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=512 --run_split=3 --num_edges=28
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=512 --run_split=4 --num_edges=138
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=512 --run_split=5 --num_edges=58
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --hidden=512 --run_split=6 --num_edges=48
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=512 --run_split=7 --num_edges=388
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.2 --dropout=0.1 --model_indices 0 --hidden=512 --run_split=8 --num_edges=38
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=512 --run_split=9 --num_edges=28

#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=citeseer --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --hidden=128 --lcc --num_edges=7358 --run_split=0
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.gcn --dataset=cora --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --hidden=512 --lcc --num_edges=10138 --run_split=0

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=CiteSeer --rewirer_step=0.1 --dropout=0.95 --model_indices 0 --hidden=128 --num_edges=8000 --runs=10 --rewired
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cora --rewirer_step=0.1 --dropout=0.95 --model_indices 0 --hidden=512 --num_edges=8000 --runs=10 --rewired

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=32 --run_split=0 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=32 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=512 --run_split=2 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=512 --run_split=3 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --hidden=1024 --run_split=4 --num_edges=754
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=1024 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=64 --run_split=6 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=64 --run_split=7 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --hidden=512 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=32 --run_split=9 --num_edges=252

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.2 --hidden=512 --run_split=1 --num_edges=2367
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.3 --hidden=512 --run_split=2 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.8 --hidden=1024 --run_split=3 --num_edges=22
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=4 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.5 --hidden=128 --run_split=6 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=1024 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=8 --num_edges=1578
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=9 --num_edges=45

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=32 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=512 --run_split=2 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=512 --run_split=3 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --hidden=1024 --run_split=4 --num_edges=754
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=1024 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --hidden=64 --run_split=6 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=64 --run_split=7 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --hidden=512 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --hidden=32 --run_split=9 --num_edges=252


#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=0 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=1 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=2 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=3 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=4 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=5 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=6 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=7 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=8 --num_edges=4063
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --hidden=256 --run_split=9 --num_edges=4063

# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=0 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=1 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=2 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=3 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=4 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=5 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=6 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=7 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=8 --num_edges=80630
# CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --hidden=256 --run_split=9 --num_edges=80630

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --hidden=2048 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --hidden=2048 --run_split=1 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --hidden=512 --run_split=2 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=512 --run_split=3 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --hidden=512 --run_split=4 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=2048 --run_split=5 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --hidden=2048 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --hidden=2048 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --hidden=512 --run_split=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --hidden=2048 --run_split=9 --num_edges=10
