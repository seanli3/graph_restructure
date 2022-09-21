echo 'edges h_den h_den_train h_den_val h_den_test h_edge h_node h_norm degree density train_acc test_acc val_acc train_acc_std test_acc_std val_acc_std'

#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --K=1 --alpha=0.1 --hidden=2048 --run_split=0 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --K=1 --alpha=0.4 --hidden=1024 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=1 --alpha=0.7 --hidden=1024 --run_split=2 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --K=1 --alpha=0.3 --hidden=1024 --run_split=3 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --K=1 --alpha=0.8 --hidden=1024 --run_split=4 --num_edges=754
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --K=1 --alpha=0.7 --hidden=1024 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.8 --model_indices 0 --K=1 --alpha=0.3 --hidden=1024 --run_split=6 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --K=1 --alpha=0.5 --hidden=1024 --run_split=7 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.7 --model_indices 0 --K=1 --alpha=0.9 --hidden=1024 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --K=1 --alpha=0.6 --hidden=1024 --run_split=9 --num_edges=252

#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.5 --K=3 --alpha=0.3 --hidden=512 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.4 --K=1 --alpha=0.8 --hidden=512 --run_split=1 --num_edges=2367
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.9 --K=1 --alpha=0.8 --hidden=512 --run_split=2 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.6 --K=1 --alpha=0.8 --hidden=512 --run_split=3 --num_edges=22
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.7 --K=1 --alpha=0.8 --hidden=512 --run_split=4 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.9 --K=1 --alpha=0.5 --hidden=512 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.8 --K=1 --alpha=0.7 --hidden=512 --run_split=6 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.9 --K=1 --alpha=0.8 --hidden=512 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.8 --K=1 --alpha=0.8 --hidden=512 --run_split=8 --num_edges=1578
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.8 --K=1 --alpha=0.8 --hidden=512 --run_split=9 --num_edges=45

#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.7 --K=3 --alpha=0.3 --hidden=1024 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.7 --K=3 --alpha=0.8 --hidden=512 --run_split=1 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.5 --K=1 --alpha=0.3 --hidden=512 --run_split=2 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.7 --K=1 --alpha=0.8 --hidden=512 --run_split=3 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.1 --K=1 --alpha=0.8 --hidden=1024 --run_split=4 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.6 --K=1 --alpha=0.3 --hidden=1024 --run_split=5 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.7 --K=1 --alpha=0.7 --hidden=1024 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.8 --K=2 --alpha=0.1 --hidden=1024 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.7 --K=1 --alpha=0.7 --hidden=512 --run_split=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Cornell --rewired --rewirer_step=0.1 --dropout=0.6 --K=1 --alpha=0.8 --hidden=1024 --run_split=9 --num_edges=10

#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=3 --alpha=0.3 --hidden=256 --run_split=0 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.2 --K=3 --alpha=0.8 --hidden=128 --run_split=1 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.3 --hidden=1024 --run_split=2 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.8 --hidden=1024 --run_split=3 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.8 --hidden=512 --run_split=4 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.3 --hidden=128 --run_split=5 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.7 --hidden=256 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=2 --alpha=0.1 --hidden=1024 --run_split=7 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.7 --hidden=256 --run_split=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.8 --hidden=512 --run_split=9 --num_edges=1

#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=0 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=1 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.2 --K=1 --alpha=0.1 --hidden=256 --run_split=2 --num_edges=520000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=512 --run_split=3 --num_edges=980000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=4 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=2 --alpha=0.1 --hidden=256 --run_split=5 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=6 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=2 --alpha=0.2 --hidden=1024 --run_split=7 --num_edges=200000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.1 --hidden=512 --run_split=8 --num_edges=420000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Squirrel --rewired --rewirer_step=0.1 --dropout=0.5 --K=1 --alpha=0.2 --hidden=128 --run_split=9 --num_edges=200000

#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=0 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=1 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.5 --K=4 --alpha=0.2 --hidden=1024 --run_split=2 --num_edges=400000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=3 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --K=8 --alpha=0.1 --hidden=1024 --run_split=4 --num_edges=1000000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.2 --K=1 --alpha=0.2 --hidden=1024 --run_split=5 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.1 --hidden=1024 --run_split=6 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=128 --run_split=7 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=8 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.appnp --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.6 --K=4 --alpha=0.2 --hidden=1024 --run_split=9 --num_edges=1000000

#for edges in `seq 1 1000 300000`
#do
#  CUDA_DEVICE=0 python -m benchmark.node_classification.appnp \
#    --dataset=Squirrel \
#    --rewired \
#    --rewirer_step=0.1 \
#    \
#    --K=1 \
#    --dropout=0.3 \
#    --alpha=0.2 \
#    --hidden=1024 \
#    --runs=5 \
#    --run_split=0 \
#    --num_edges=$edges\
#    --max_node_degree=5000
#done
