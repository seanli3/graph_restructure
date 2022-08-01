echo "GAT Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Wisconsin --dropout=0.4 --heads=8 --hidden=128 --run_split=9

echo "GAT Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=12 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Texas --dropout=0.7 --heads=10 --hidden=512 --run_split=9

echo "GAT Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Cornell --dropout=0.5 --heads=8 --hidden=512 --run_split=9

echo "GAT Actor"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=0 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.2 --hidden=256 --run_split=1 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=2 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=3 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=4 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=5 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=6 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=7 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=8 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Actor --dropout=0.3 --hidden=256 --run_split=9 --heads=8

echo "GAT Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=0 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=1 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=2 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=3 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=4 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=5 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=6 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=7 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=8 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Squirrel --dropout=0.3 --hidden=256 --run_split=9 --heads=8

echo "GAT Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=0 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=1 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=2 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=3 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=4 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=5 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=6 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=7 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=8 --heads=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gat --dataset=Chameleon --dropout=0.5 --hidden=64 --run_split=9 --heads=8
