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
CUDA_DEVICE=2 python -m benchmark.node_classification.arma --dataset=actor --rewirer_step=0.1 --dropout=0.9 --model_indices 0 --num_edges=0 --num_stacks=2 --num_layers=1 --hidden=128 --skip_dropout=0.3