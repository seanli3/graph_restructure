echo 'dataset,split,dropout,hidden,edge_step,rewirer_step,edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std'

CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=0 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.7 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=1 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=2 --edge_step=5
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.7 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=3 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=4 --edge_step=5
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=5 --edge_step=50
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.7 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=6 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=7 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=8 --edge_step=50
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=9 --edge_step=5

CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.9 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=0 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.4 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=1 --edge_step=50
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.7 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=2 --edge_step=50
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=3 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=4 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.9 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=5 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.5 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=6 --edge_step=200
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=7 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.5 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=8 --edge_step=50
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Texas --rewired --rewirer_step=0.2 --dropout=0.9 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=9 --edge_step=1

CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.2 --dropout=0.8 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=0 --edge_step=50
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.5 --eps=0.05 --with_rand_signal --with_node_feature --hidden=512 --run_split=1 --edge_step=5
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.9 --eps=0.05 --with_rand_signal --with_node_feature --hidden=512 --run_split=2 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.7 --eps=0.05 --with_rand_signal --with_node_feature --hidden=1024 --run_split=3 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.7 --eps=0.05 --with_rand_signal --with_node_feature --hidden=1024 --run_split=4 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.4 --eps=0.05 --with_rand_signal --with_node_feature --hidden=1024 --run_split=5 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.4 --dropout=0.4 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=6 --edge_step=5
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.5 --eps=0.05 --with_rand_signal --with_node_feature --hidden=1024 --run_split=7 --edge_step=1
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.05 --dropout=0.8 --eps=0.05 --with_rand_signal --with_node_feature --hidden=512 --run_split=8 --edge_step=10
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Cornell --rewired --rewirer_step=0.5 --dropout=0.9 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=9 --edge_step=10

CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=0 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=1 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=2 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=3 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=64 --run_split=4 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.4 --eps=0.1 --with_rand_signal --with_node_feature --hidden=128 --run_split=5 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.4 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=6 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=1024 --run_split=7 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=256 --run_split=8 --edge_step=100
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Actor --rewired --rewirer_step=0.1 --dropout=0.3 --eps=0.1 --with_rand_signal --with_node_feature --hidden=512 --run_split=9 --edge_step=100

CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.7 --dropout=0.4 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=0 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.6 --dropout=0.6 --eps=0.2 --with_rand_signal --with_node_feature --hidden=512 --run_split=1 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.15 --dropout=0.3 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=2 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.15 --dropout=0.7 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=3 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.15 --dropout=0.5 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=4 --edge_step=10000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.3 --dropout=0.5 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=5 --edge_step=10000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.7 --dropout=0.3 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=6 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.6 --dropout=0.7 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=7 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.6 --dropout=0.5 --eps=0.2 --with_rand_signal --with_node_feature --hidden=1024 --run_split=8 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Squirrel --rewired --rewirer_step=0.7 --dropout=0.5 --eps=0.2 --with_rand_signal --with_node_feature --hidden=512 --run_split=9 --edge_step=10000

CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.5 --dropout=0.2 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=0 --edge_step=10000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.5 --dropout=0.6 --eps=0.2 --with_rand_signal --with_node_feature --hidden=512 --run_split=1 --edge_step=5000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.5 --dropout=0.3 --eps=0.2 --with_rand_signal --with_node_feature --hidden=512 --run_split=2 --edge_step=2000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.3 --dropout=0.7 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=3 --edge_step=1000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.5 --dropout=0.4 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=4 --edge_step=20000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.25 --dropout=0.8 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=5 --edge_step=10000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.15 --dropout=0.6 --eps=0.2 --with_rand_signal --with_node_feature --hidden=512 --run_split=6 --edge_step=2000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.1 --dropout=0.8 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=7 --edge_step=1000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.4 --dropout=0.6 --eps=0.2 --with_rand_signal --with_node_feature --hidden=256 --run_split=8 --edge_step=5000
CUDA_DEVICE=1 python -m benchmark.node_classification.gcn --dataset=Chameleon --rewired --rewirer_step=0.6 --dropout=0.3 --eps=0.2 --with_rand_signal --with_node_feature --hidden=512 --run_split=9 --edge_step=5000
#
#for dataset in "Texas"
#do
#  for split in 0 1 2 3 4 5 6 7 8 9
#  do
#    for dropout in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
#    do
#      for hidden in 256 512 1024
#      do
#        for step in 1 5 10 50
#        do
#          for rewirer_step in 0.2
#          do
#            echo -n "$dataset,$split,$dropout,$hidden,$step,"
#            CUDA_DEVICE=1 python -m benchmark.node_classification.gcn \
#              --dataset=$dataset \
#              --rewired \
#              --rewirer_step=$rewirer_step \
#              --dropout=$dropout \
#              --hidden=$hidden \
#              --run_split=$split \
#              --edge_step=$step \
#              --with_rand_signal \
#              --with_node_feature \
#              --self_loop \
#              --eps=0.1
#          done
#        done
#      done
#    done
#  done
#done
