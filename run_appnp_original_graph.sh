echo "APPNP Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.3 --K=1 --alpha=0.1 --hidden=2048 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.1 --K=1 --alpha=0.4 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.2 --K=1 --alpha=0.7 --hidden=1024 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.4 --K=1 --alpha=0.3 --hidden=1024 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.1 --K=1 --alpha=0.8 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.2 --K=1 --alpha=0.7 --hidden=1024 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.8 --K=1 --alpha=0.3 --hidden=1024 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.1 --K=1 --alpha=0.5 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.7 --K=1 --alpha=0.9 --hidden=1024 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Wisconsin  --dropout=0.1 --K=1 --alpha=0.6 --hidden=1024 --run_split=9

echo "APPNP Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.5 --K=3 --alpha=0.3 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.4 --K=1 --alpha=0.8 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.9 --K=1 --alpha=0.8 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.6 --K=1 --alpha=0.8 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.7 --K=1 --alpha=0.8 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.9 --K=1 --alpha=0.5 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.8 --K=1 --alpha=0.7 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.9 --K=1 --alpha=0.8 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.8 --K=1 --alpha=0.8 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Texas --dropout=0.8 --K=1 --alpha=0.8 --hidden=512 --run_split=9

echo "APPNP Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.7 --K=3 --alpha=0.3 --hidden=1024 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.7 --K=3 --alpha=0.8 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.5 --K=1 --alpha=0.3 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.7 --K=1 --alpha=0.8 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.1 --K=1 --alpha=0.8 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.6 --K=1 --alpha=0.3 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.7 --K=1 --alpha=0.7 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.8 --K=2 --alpha=0.1 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.7 --K=1 --alpha=0.7 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Cornell --dropout=0.6 --K=1 --alpha=0.8 --hidden=512 --run_split=9

echo "APPNP Actor"
echo "train_acc,test_acc,val_acc,train512_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=3 --alpha=0.3 --hidden=256 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.2 --K=3 --alpha=0.8 --hidden=256 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.3 --hidden=256 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.8 --hidden=256 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.8 --hidden=256 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.3 --hidden=256 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.7 --hidden=256 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=2 --alpha=0.1 --hidden=256 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.7 --hidden=256 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Actor --dropout=0.3 --K=1 --alpha=0.8 --hidden=256 --run_split=9

echo "APPNP Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=1 --alpha=0.2 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=1 --alpha=0.2 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.2 --K=1 --alpha=0.1 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=1 --alpha=0.2 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=1 --alpha=0.2 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=2 --alpha=0.1 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=1 --alpha=0.2 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=2 --alpha=0.2 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.3 --K=1 --alpha=0.1 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Squirrel --dropout=0.5 --K=1 --alpha=0.2 --hidden=512 --run_split=9

echo "APPNP Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.5 --K=4 --alpha=0.2 --hidden=1024 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.6 --K=8 --alpha=0.1 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.2 --K=1 --alpha=0.2 --hidden=1024 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.3 --K=1 --alpha=0.1 --hidden=1024 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.3 --K=1 --alpha=0.2 --hidden=1024 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.appnp --dataset=Chameleon --dropout=0.6 --K=4 --alpha=0.2 --hidden=1024 --run_split=9
