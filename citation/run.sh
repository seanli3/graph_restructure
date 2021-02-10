#!/bin/sh

runs=10

echo "Cora"
echo "===="
#
echo "GCN"
python gcn.py --dataset=Cora --runs=$runs --split=public --cuda
##python gcn.py --dataset=Cora --random_splits=True

echo "GAT"
python gat.py --dataset=Cora --runs=$runs --split=public --cuda
##python gat.py --dataset=Cora --random_splits=True
#
echo "Cheby"
python cheb.py --dataset=Cora --num_hops=3 --runs=$runs --split=public --cuda
##python cheb.py --dataset=Cora --num_hops=3 --random_splits=True
#
echo "SGC"
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --runs=$runs --split=public --cuda
##python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --random_splits=True
#
echo "ARMA"
python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --runs=$runs --split=public --cuda
##python arma.py --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True
#
echo "APPNP"
python appnp.py --dataset=Cora --alpha=0.1 --runs=$runs --split=public --cuda
##python appnp.py --dataset=Cora --alpha=0.1 --random_splits=True

echo "CiteSeer"
echo "========"

echo "GCN"
python gcn.py --dataset=CiteSeer --runs=$runs --split=public --cuda
#python gcn.py --dataset=CiteSeer --random_splits=True

echo "GAT"
python gat.py --dataset=CiteSeer --runs=$runs --split=public --cuda
#python gat.py --dataset=CiteSeer --random_splits=True

echo "Cheby"
python cheb.py --dataset=CiteSeer --num_hops=2 --runs=$runs --split=public --cuda
#python cheb.py --dataset=CiteSeer --num_hops=3 --random_splits=True

echo "SGC"
python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --runs=$runs --split=public --cuda
#python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --random_splits=True

echo "ARMA"
python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --runs=$runs --cuda
#python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --random_splits=True

echo "APPNP"
python appnp.py --dataset=CiteSeer --alpha=0.1 --runs=$runs --split=public --cuda
#python appnp.py --dataset=CiteSeer --alpha=0.1 --random_splits=True

echo "PubMed"
echo "======"

echo "GCN"
python gcn.py --dataset=PubMed --runs=$runs --split=public --cuda
#python gcn.py --dataset=PubMed --random_splits=True

echo "GAT"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --runs=$runs --split=public --cuda
#python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits=True

echo "Cheby"
python cheb.py --dataset=PubMed --num_hops=2 --runs=$runs --split=public --cuda
#python cheb.py --dataset=PubMed --num_hops=2 --random_splits=True

echo "SGC"
python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --runs=$runs --split=public --cuda
#python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --random_splits=True

echo "ARMA"
python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0 --runs=$runs --split=public --cuda
##python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits=True

echo "APPNP"
python appnp.py --dataset=PubMed --alpha=0.1 --runs=$runs --split=public --cuda
#python appnp.py --dataset=PubMed --alpha=0.1 --random_splits=True
