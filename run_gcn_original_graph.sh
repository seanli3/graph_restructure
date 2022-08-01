echo "GCN Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Wisconsin --dropout=0.1 --hidden=512 --run_split=9

echo "GCN Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Texas --dropout=0.9 --hidden=512 --run_split=9

echo "GCN Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Cornell --dropout=0.8 --hidden=512 --run_split=9

echo "GCN Actor"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Actor --dropout=0.3 --hidden=256 --run_split=9

echo "GCN Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Squirrel --dropout=0.3 --hidden=1024 --run_split=9

echo "GCN Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gcn --dataset=Chameleon --dropout=0.3 --hidden=1024 --run_split=9
