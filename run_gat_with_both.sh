echo "GAT Wisconsin"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --heads=8 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --heads=8 --hidden=128 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=128 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --heads=8 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.4 --model_indices 0 --heads=8 --hidden=128 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=256 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --heads=8 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.2 --model_indices 0 --heads=8 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --heads=8 --hidden=256 --run_split=9

echo "GAT Texas"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --heads=10 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --heads=10 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.5 --heads=10 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=10 --hidden=256 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=12 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=10 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --heads=10 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --heads=10 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --heads=10 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --heads=10 --hidden=512 --run_split=9

echo "GAT Cornell"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --heads=12 --hidden=2048 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --heads=8 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --heads=8 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --heads=8 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.8 --heads=8 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --heads=8 --hidden=2048 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --heads=10 --hidden=2048 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --heads=8 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --heads=8 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --heads=8 --hidden=1024 --run_split=9

echo "GAT Actor"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=0 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=1 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=2 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=3 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=4 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=5 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=6 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=7 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=8 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat  --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=7600 --hidden=1024 --run_split=9 --heads=8

echo "GAT Squirrel"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=10000 --hidden=1024 --run_split=0 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --edge_step=10000 --hidden=1024 --run_split=1 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --edge_step=10000 --hidden=512 --run_split=2 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --edge_step=10000 --hidden=512 --run_split=3 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --edge_step=10000 --hidden=32 --run_split=4 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --edge_step=10000 --hidden=128 --run_split=5 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --edge_step=10000 --hidden=128 --run_split=6 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=10000 --hidden=1024 --run_split=7 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=10000 --hidden=256 --run_split=8 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --edge_step=10000 --hidden=512 --run_split=9 --heads=8

echo "GAT Chameleon"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --edge_step=10000 --hidden=64 --run_split=0 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --edge_step=10000 --hidden=64 --run_split=1 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --edge_step=10000 --hidden=64 --run_split=2 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --edge_step=10000 --hidden=1024 --run_split=3 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.1 --edge_step=10000 --hidden=64 --run_split=4 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --edge_step=10000 --hidden=64 --run_split=5 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --edge_step=10000 --hidden=64 --run_split=6 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --edge_step=10000 --hidden=256 --run_split=7 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --edge_step=10000 --hidden=128 --run_split=8 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --edge_step=10000 --hidden=64 --run_split=9 --heads=8
