echo 'edges h_den h_den_train h_den_val h_den_test h_edge h_node h_norm degree density train_acc test_acc val_acc train_acc_std test_acc_std val_acc_std'

CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=0 --num_edges=32
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=1 --num_edges=1005
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=2 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=3 --num_edges=1509
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=4 --num_edges=1761
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=5 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=6 --num_edges=503
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=7 --num_edges=503
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=8 --num_edges=252
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=9 --num_edges=1761

#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=0 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=1 --num_edges=2367
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=2 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=3 --num_edges=22
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=4 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=5 --num_edges=1315
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=6 --num_edges=516
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=8 --num_edges=1578
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=9 --num_edges=45

#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=0 --num_edges=2104
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=1 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=2 --num_edges=263
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=3 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=4 --num_edges=90
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=5 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=7 --num_edges=789
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=8 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=9 --num_edges=10

#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=0 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=1 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=2 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=3 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=4 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=5 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=6 --num_edges=10
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=7 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=8 --num_edges=1
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=9 --num_edges=1


#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=0 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=1 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=2 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=3 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=4 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=5 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=6 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=7 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=8 --num_edges=300000 --max_node_degree=5
#CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=9 --num_edges=300000 --max_node_degree=5

#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=0 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=1 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=2 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --run_split=3 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=4 --num_edges=300000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=5 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=6 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=7 --num_edges=1000000 --max_node_degree=15
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=8 --num_edges=800000 --max_node_degree=10
#CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=9 --num_edges=800000 --max_node_degree=10

#for edges in `seq 1 500 300000`
#do
#  CUDA_DEVICE=0 python -m benchmark.node_classification.sgc \
#    --dataset=Squirrel \
#    --rewired \
#    --rewirer_step=0.1 \
#    --model_indices 0 \
#    --K=2 \
#    --runs=5 \
#    --run_split=0 \
#    --num_edges=$edges\
#    --max_node_degree=5000
#done
