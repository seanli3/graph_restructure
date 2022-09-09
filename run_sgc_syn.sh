echo "SGC Cora Synthetic"

echo "h_den,train_acc,test_acc,val_acc,train_acc_std,test_acc_std,val_acc_std"
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.9 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.8 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.7 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.6 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.53 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.52 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.51 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.504 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.503 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.5025 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.502 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.5015 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.501 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.5005 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.5 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.499 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.498 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.497 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.496 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.495 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.49 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.48 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.47 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.46 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.45 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.44 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.43 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.42 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.41 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.4 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.3 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.2 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0.1 --runs=10
CUDA_DEVICE=1 python -m benchmark.node_classification.sgc --dataset=Cora --rewirer_step=0.1 --K=3 --h_den=0 --runs=10
