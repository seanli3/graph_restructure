echo 'edges h_den h_den_train h_den_val h_den_test h_edge h_node h_norm degree density train_acc test_acc val_acc train_acc_std test_acc_std val_acc_std'

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

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=256 --run_split=0 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --num_hops=1 --hidden=128 --run_split=1 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=1024 --run_split=2 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=1024 --run_split=3 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=512 --run_split=4 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=128 --run_split=5 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=256 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=1024 --run_split=7 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=256 --run_split=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_hops=1 --hidden=512 --run_split=9 --num_edges=1

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=128 --run_split=0 --num_edges=200000 --max_node_degree=4
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=64 --run_split=1 --num_edges=230000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=32 --run_split=2 --num_edges=220000 --max_node_degree=12
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=64 --run_split=3 --num_edges=400000 --max_node_degree=16
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=128 --run_split=4 --num_edges=540000 --max_node_degree=6
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=128 --run_split=5 --num_edges=400000 --max_node_degree=4
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=128 --run_split=6 --num_edges=260000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=64 --run_split=7 --num_edges=160000 --max_node_degree=12
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=32 --run_split=8 --num_edges=200000 --max_node_degree=14
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=32 --run_split=9 --num_edges=220000 --max_node_degree=14

#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=0 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=1 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=128 --run_split=2 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=3 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=1024 --run_split=4 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=5 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=512 --run_split=6 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=256 --run_split=7 --num_edges=400000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=1024 --run_split=8 --num_edges=600000 --max_node_degree=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.cheb --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_hops=2 --hidden=32 --run_split=9 --num_edges=500000 --max_node_degree=10

