#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=34277 --run_split=9 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=34277 --run_split=8 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=26277 --run_split=7 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=42277 --run_split=6 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=46277 --run_split=5 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=30277 --run_split=4 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=42277 --run_split=3 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=46277 --run_split=2 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=30277 --run_split=1 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=30277 --run_split=0 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75

#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_edges=30277 --num_stacks=3 --num_layers=1 --hidden=128 --skip_dropout=0.75
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=squirrel --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --num_edges=17336 --num_stacks=4 --num_layers=1 --hidden=128 --skip_dropout=0.5
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --num_edges=16076 --num_stacks=4 --num_layers=1 --hidden=128 --skip_dropout=0.5 --runs=10
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.arma --dataset=cora --rewired --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=99744 --num_stacks=5 --num_layers=1 --hidden=128 --skip_dropout=0.5 --runs=10
#CPU_ONLY=1 python -m benchmark.node_classification.arma --dataset=actor --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --num_edges=0 --num_stacks=1 --num_layers=1 --hidden=128 --skip_dropout=0.3
#CUDA_DEVICE=2 python -m benchmark.node_classification.arma --dataset=actor --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=0 --num_stacks=2 --num_layers=1 --hidden=128 --skip_dropout=0.3

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=CiteSeer --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --num_stacks=2 --hidden=512 --num_edges=7800 --runs=10 --num_layers=1 --skip_dropout=0.3
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cora --rewirer_step=0.1 --model_indices 0 --dropout=0.9  --num_stacks=6 --hidden=512 --num_edges=7800 --runs=10 --num_layers=1 --skip_dropout=0.6

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_stacks=2 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=0 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=64 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=2 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=64 --run_split=3 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=4 --num_edges=754
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=32 --run_split=6 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=7 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=9 --num_edges=252

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=1 --num_edges=2367
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.4 --hidden=512 --run_split=2 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.8 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=3 --num_edges=22
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=4 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.8 --hidden=512 --run_split=6 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.1 --hidden=512 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.5 --hidden=128 --run_split=8 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --num_stacks=1 --num_layers=1 --skip_dropout=0.5 --hidden=512 --run_split=9 --num_edges=45

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=128 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=1 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.2 --hidden=512 --run_split=2 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=3 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=4 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.4 --hidden=512 --run_split=5 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --num_stacks=1 --num_layers=1 --skip_dropout=0.8 --hidden=512 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.2 --hidden=512 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=9 --num_edges=10

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=0 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=1 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=2 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=3 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=4 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=5 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=7 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=9 --num_edges=1

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=0 --num_edges=500000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=1 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.4 --hidden=64 --run_split=2 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.4 --hidden=256 --run_split=3 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=4 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=128 --run_split=5 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=6 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=7 --num_edges=400000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.5 --hidden=256 --run_split=8 --num_edges=400000 --max_node_degree=8
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=9 --num_edges=400000 --max_node_degree=8

#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=0 --num_edges=900000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=1 --num_edges=900000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=2 --num_edges=900000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=3 --num_edges=900000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=4 --num_edges=500000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=5 --num_edges=900000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=6 --num_edges=900000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=7 --num_edges=1000000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=8 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.arma --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=9 --num_edges=900000 --max_node_degree=10
