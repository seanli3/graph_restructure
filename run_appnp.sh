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
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=citeseer --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --num_edges=16076 --hidden=256 --K=2 --alpha=0.1 --runs=10
#CUDA_DEVICE=6 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cora --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_edges=99744 --hidden=512 --K=1 --alpha=0.3 --runs=10
#CPU_ONLY=1 python -m benchmark.node_classification.appnp --dataset=actor --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_edges=0 --hidden=128 --K=2 --alpha=0.3
#CPU_ONLY=1 python -m benchmark.node_classification.appnp --dataset=actor --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_edges=0 --hidden=128 --K=2 --alpha=0.3

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=5278 --hidden=32 --run_split=0
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=3938 --hidden=32 --run_split=1
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=4478 --hidden=32 --run_split=2
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=38 --hidden=32 --run_split=3
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=1308 --hidden=32 --run_split=4
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=28 --hidden=32 --run_split=5
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=18 --hidden=32 --run_split=6
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=808 --hidden=32 --run_split=7
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=78 --hidden=32 --run_split=8
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --num_edges=3558 --hidden=32 --run_split=9
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=texas --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --K=10 --alpha=0.1 --num_edges=3558 --hidden=512

#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --K=1 --alpha=0.3 --hidden=512 --run_split=0 --num_edges=7835
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --K=1 --alpha=0.4 --hidden=512 --run_split=1 --num_edges=1195
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=4 --alpha=0.1 --hidden=512 --run_split=2 --num_edges=2145
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=3 --num_edges=9045
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=4 --num_edges=1085
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=5 --num_edges=2405
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=6 --num_edges=65
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=7 --num_edges=135
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=8 --num_edges=685
#CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=9 --num_edges=115

CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --K=1 --alpha=0.3 --hidden=512 --run_split=0  --num_edges=168
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --K=1 --alpha=0.4 --hidden=512 --run_split=1  --num_edges=28
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=4 --alpha=0.1 --hidden=512 --run_split=2  --num_edges=28
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=3  --num_edges=28
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=4  --num_edges=138
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=5  --num_edges=58
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=6  --num_edges=48
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=7  --num_edges=388
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=8  --num_edges=38
CUDA_DEVICE=2 /data_seoul/seanl/miniconda/envs/rewire/bin/python -m benchmark.node_classification.appnp --dataset=cornell --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=3 --alpha=0.1 --hidden=32 --run_split=9  --num_edges=28
