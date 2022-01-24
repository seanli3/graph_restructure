#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=34277 --run_split=9 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=34277 --run_split=8 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=26277 --run_split=7 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=42277 --run_split=6 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=46277 --run_split=5 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=30277 --run_split=4 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=42277 --run_split=3 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=46277 --run_split=2 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=30277 --run_split=1 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=30277 --run_split=0 --K=2

#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=chameleon --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=30277 --K=2
#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=squirrel --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=17336 --K=1

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=citeseer --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=16076 --K=3 --runs=10
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.sgc --dataset=cora --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=99744 --K=1 --runs=10
#CPU_ONLY=1 python -m benchmark.node_classification.sgc --dataset=actor --rewired --rewirer_step=0.1 --model_indices 0 --num_edges=0 --K=2
CPU_ONLY=1 python -m benchmark.node_classification.sgc --dataset=actor --rewirer_step=0.1 --model_indices 0 --num_edges=0 --K=2
