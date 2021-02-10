#!/bin/sh

dissimilar_t=0.8
runs=10

echo "Cora"
echo "===="

echo "GCN"
python gcn.py --dataset=Cora --dissimilar_t=$dissimilar_t  --runs=$runs
#python gcn.py --dataset=Cora

echo "GAT"
python gat.py --dataset=Cora --dissimilar_t=$dissimilar_t  --runs=$runs
#python gat.py --dataset=Cora

echo "Cheby"
python cheb.py --dataset=Cora --num_hops=3 --dissimilar_t=$dissimilar_t  --runs=$runs
#python cheb.py --dataset=Cora --num_hops=3

echo "SGC"
python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --dissimilar_t=$dissimilar_t  --runs=$runs
#python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005

echo "ARMA"
python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --dissimilar_t=$dissimilar_t  --runs=$runs
#python arma.py --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights=True

echo "APPNP"
python appnp.py --dataset=Cora --alpha=0.1 --dissimilar_t=$dissimilar_t  --runs=$runs
#python appnp.py --dataset=Cora --alpha=0.1

#echo "CiteSeer"
#echo "========"
#
#echo "GCN"
#python gcn.py --dataset=Citeseer --dissimilar_t=$dissimilar_t
##python gcn.py --dataset=CiteSeer
#
#echo "GAT"
#python gat.py --dataset=Citeseer --dissimilar_t=$dissimilar_t
##python gat.py --dataset=CiteSeer
#
#echo "Cheby"
#python cheb.py --dataset=Citeseer --num_hops=2 --dissimilar_t=$dissimilar_t
##python cheb.py --dataset=CiteSeer --num_hops=3
#
#echo "SGC"
#python sgc.py --dataset=Citeseer --K=2 --weight_decay=0.005 --dissimilar_t=$dissimilar_t
##python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005
#
#echo "ARMA"
#python arma.py --dataset=Citeseer --num_stacks=3 --num_layers=1 --shared_weights=True --dissimilar_t=$dissimilar_t
##python arma.py --dataset=Citeseer --num_stacks=3 --num_layers=1 --shared_weights=True
#
#echo "APPNP"
#python appnp.py --dataset=Citeseer --alpha=0.1 --dissimilar_t=$dissimilar_t
##python appnp.py --dataset=Citeseer --alpha=0.1
#
#echo "PubMed"
#echo "======"
#
#echo "GCN"
#python gcn.py --dataset=PubMed --dissimilar_t=$dissimilar_t
##python gcn.py --dataset=PubMed
#
#echo "GAT"
#python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --dissimilar_t=$dissimilar_t
##python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8
#
#echo "Cheby"
#python cheb.py --dataset=PubMed --num_hops=2 --dissimilar_t=$dissimilar_t
##python cheb.py --dataset=PubMed --num_hops=2
#
#echo "SGC"
#python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --dissimilar_t=$dissimilar_t
##python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005
#
#echo "ARMA"
#python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0 --dissimilar_t=$dissimilar_t
##python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5
#
#echo "APPNP"
#python appnp.py --dataset=PubMed --alpha=0.1 --dissimilar_t=$dissimilar_t
##python appnp.py --dataset=PubMed --alpha=0.1
