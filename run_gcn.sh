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

CPU_ONLY=1 python -m benchmark.node_classification.gcn --dataset=actor --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=0 --hidden=256
