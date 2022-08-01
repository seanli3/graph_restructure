echo "CHEB Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Wisconsin  --dropout=0.6 --num_hops=2 --hidden=512 --run_split=9

echo "CHEB Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Texas --dropout=0.5 --num_hops=2 --hidden=512 --run_split=9

echo "CHEB Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Cornell --dropout=0.8 --num_hops=2 --hidden=1024 --run_split=9

echo "CHEB Actor"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Actor --dropout=0.3 --num_hops=1 --hidden=256 --run_split=9

echo "CHEB Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Squirrel --dropout=0.9 --num_hops=2 --hidden=128 --run_split=9

echo "CHEB Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.cheb --dataset=Chameleon --dropout=0.9 --num_hops=2 --hidden=512 --run_split=9

