echo "SGC Wisconsin"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Wisconsin --K=1 --run_split=9

echo "SGC Texas"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Texas --K=1 --run_split=9

echo "SGC Cornell"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Cornell --K=1 --run_split=9

echo "SGC Actor"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Actor --K=1 --run_split=9

echo "SGC Squirrel"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Squirrel --K=2 --run_split=9

echo "SGC Chameleon"
echo "train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=0
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=1
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=2
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=2 --run_split=3
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=4
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=5
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=6
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=7
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=8
CUDA_DEVICE=0 python -m benchmark.node_classification.sgc --dataset=Chameleon --K=1 --run_split=9
