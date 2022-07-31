echo "ARMA Wisconsin"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_stacks=2 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=0
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=64 --run_split=1
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.6 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=2
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.3 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=64 --run_split=3
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=4
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=32 --run_split=6
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=7
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.5 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=8
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Wisconsin --rewired --rewirer_step=0.1 --dropout=0.1 --model_indices 0 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=32 --run_split=9

echo "ARMA Texas"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=0
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=1
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.4 --hidden=512 --run_split=2
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.8 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=3
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=4
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.8 --hidden=512 --run_split=6
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.1 --hidden=512 --run_split=7
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.5 --hidden=128 --run_split=8
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Texas --rewired --rewirer_step=0.2 --model_indices 0 --dropout=0.7 --num_stacks=1 --num_layers=1 --skip_dropout=0.5 --hidden=512 --run_split=9

echo "ARMA Cornell"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=128 --run_split=0
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=1
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.2 --hidden=512 --run_split=2
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=3
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.7 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=4
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.4 --hidden=512 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --num_stacks=1 --num_layers=1 --skip_dropout=0.8 --hidden=512 --run_split=6
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.2 --hidden=512 --run_split=7
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=8
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Cornell --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=9

echo "ARMA Actor"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=0
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.2 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=1
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=2
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=3
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=4
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=6
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=1024 --run_split=7
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=8
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Actor --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=9

echo "ARMA Squirrel"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=0
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=1
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.4 --hidden=64 --run_split=2
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.4 --hidden=256 --run_split=3
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=4
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=128 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=6
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=7
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.5 --hidden=256 --run_split=8
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Squirrel --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=9

echo "ARMA Chameleon"
echo "edges,h_den,h_den_train,h_den_val,h_den_test,h_edge,h_node,h_norm,degree,density,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=0
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=1
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=2
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=3
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=4
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=5
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=6
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=7
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=8
CUDA_DEVICE=1 python -m benchmark.node_classification.arma --with_rand_signal --dataset=Chameleon --rewired --rewirer_step=0.1 --model_indices 0 --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=9
