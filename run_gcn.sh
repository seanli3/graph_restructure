echo 'edges h_den h_den_train h_den_val h_den_test h_edge h_node h_norm degree density train_acc test_acc val_acc train_acc_std test_acc_std val_acc_std'

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
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=4 --num_edges=20
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.8 --hidden=512 --run_split=6 --num_edges=20
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=1024 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=8 --num_edges=1578
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --hidden=512 --run_split=9 --num_edges=45

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

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=0 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=1 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=2 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=3 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=64 --run_split=4 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=128 --run_split=5 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=512 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=7 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=512 --run_split=9 --num_edges=1

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=0 --num_edges=1000000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=1 --num_edges=1000000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=2 --num_edges=500000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --hidden=1024 --run_split=3 --num_edges=100000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=32 --run_split=4 --num_edges=1000000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=5 --num_edges=900000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --hidden=1024 --run_split=6 --num_edges=400000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=7 --num_edges=1000000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --hidden=128 --run_split=8 --num_edges=500000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=1024 --run_split=9 --num_edges=1000000

#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=0 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=1 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=2 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --hidden=128 --run_split=3 --num_edges=900000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --hidden=512 --run_split=4 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=5 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=6 --num_edges=700000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --hidden=1024 --run_split=7 --num_edges=400000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=256 --run_split=8 --num_edges=400000
#CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --hidden=1024 --run_split=9 --num_edges=700000

#
#echo 'edges h_den h_den_train h_den_val h_den_test h_edge h_node h_norm degree density train_acc test_acc val_acc train_acc_std test_acc_std val_acc_std'
#for edges in `seq 1 1000 2000000`
#do
#  CUDA_DEVICE=0 python -m benchmark.node_classification.gcn \
#    --dataset=Squirrel\
#    --rewired \
#    --rewirer_step=0.1 \
#    --model_indices 0 \
#    --dropout=0.3 \
#    --runs=5 \
#    --hidden=1024 \
#    --run_split=0 \
#    --num_edges=$edges\
#    --max_node_degree=5000
#  done
