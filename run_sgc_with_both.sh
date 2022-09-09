echo "SGC Wisconsin"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=9

echo "SGC Texas"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --K=1 --run_split=9

echo "SGC Cornell"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --run_split=9

echo "SGC Actor"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=7600 --run_split=9

echo "SGC Squirrel"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=9

echo "SGC Chameleon"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=2 --edge_step=10000 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --with_rand_signal --with_node_feature --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --K=1 --edge_step=10000 --run_split=9
