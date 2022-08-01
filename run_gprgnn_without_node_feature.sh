echo "GAT Wisconsin"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=9


echo "GAT Texas"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices=0 --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=9

echo "GAT Cornell"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=9

echo "GAT Actor"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=9

echo "GAT Squirrel"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices=0 --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=9

echo "GAT Chameleon"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices=0 --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=9