echo "ARMA Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.5 --num_stacks=2 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=128 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Wisconsin  --dropout=0.1 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=128 --run_split=9

echo "ARMA Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.4 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.8 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.8 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.1 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.5 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Texas --dropout=0.7 --num_stacks=1 --num_layers=1 --skip_dropout=0.5 --hidden=512 --run_split=9

echo "ARMA Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.6 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.2 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.9 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.7 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.4 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.2 --num_stacks=1 --num_layers=1 --skip_dropout=0.8 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.4 --num_stacks=1 --num_layers=1 --skip_dropout=0.2 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Cornell --dropout=0.6 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=512 --run_split=9

echo "ARMA Actor"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.2 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.5 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Actor --dropout=0.3 --num_stacks=1 --num_layers=1 --skip_dropout=0.3 --hidden=256 --run_split=9

echo "ARMA Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.4 --hidden=256 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.4 --hidden=256 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.5 --hidden=256 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Squirrel --dropout=0.9 --num_stacks=4 --num_layers=1 --skip_dropout=0.6 --hidden=256 --run_split=9

echo "ARMA Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.arma --dataset=Chameleon --dropout=0.9 --num_stacks=5 --num_layers=1 --skip_dropout=0.6 --hidden=1024 --run_split=9
