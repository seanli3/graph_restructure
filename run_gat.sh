echo 'edges h_den h_den_train h_den_val h_den_test h_edge h_node h_norm degree density train_acc test_acc val_acc train_acc_std test_acc_std val_acc_std'

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

#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=0 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --hidden=256 --run_split=1 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=2 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=3 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=64 --run_split=4 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=128 --run_split=5 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=512 --run_split=6 --heads=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=7 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=8 --heads=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=512 --run_split=9 --heads=8 --num_edges=1

#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=0 --heads=8 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --hidden=1024 --run_split=1 --heads=8 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=512 --run_split=2 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=512 --run_split=3 --heads=8 --num_edges=600000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --hidden=32 --run_split=4 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=128 --run_split=5 --heads=8 --num_edges=600000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --hidden=128 --run_split=6 --heads=8 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=7 --heads=8 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=8 --heads=8 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --hidden=512 --run_split=9 --heads=8 --num_edges=200000

#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=64 --run_split=0 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=64 --run_split=1 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=64 --run_split=2 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --hidden=1024 --run_split=3 --heads=8 --num_edges=800000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --hidden=64 --run_split=4 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=64 --run_split=5 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=64 --run_split=6 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=7 --heads=8 --num_edges=900000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=128 --run_split=8 --heads=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gat --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --hidden=64 --run_split=9 --heads=8 --num_edges=800000
