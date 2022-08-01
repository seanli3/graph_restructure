echo "GPRGNN Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Wisconsin --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=9


echo "GPRGNN Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Texas --alpha=1 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=9

echo "GPRGNN Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Cornell --alpha=0.9 --weight_decay=0.0005 --lr=0.05 --dropout=0.5 --hidden=64 --run_split=9

echo "GPRGNN Actor"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Actor --alpha=1 --weight_decay=0 --lr=0.01 --dprate=0.9 --dropout=0.5 --hidden=64 --run_split=9

echo "GPRGNN Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Squirrel --alpha=0 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=9

echo "GPRGNN Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.gprgnn --dataset=Chameleon --alpha=1 --weight_decay=0 --lr=0.05 --dprate=0.7 --dropout=0.5 --hidden=64 --run_split=9