#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=34277 --run_split=9
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=34277 --run_split=8
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=26277 --run_split=7
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=42277 --run_split=6
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=46277 --run_split=5
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=30277 --run_split=4
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=42277 --run_split=3
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=46277 --run_split=2
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=30277 --run_split=1
#CUDA_DEVICE=1 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=30277 --run_split=0

#CUDA_DEVICE=0 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --num_edges=30277

#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --num_edges=17336 --hidden=512 --K=2 --alpha=0.1
CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --num_edges=16076 --hidden=256 --K=2 --alpha=0.1 --runs=10



