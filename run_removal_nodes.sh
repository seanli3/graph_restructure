#!/bin/sh

echo "CiteSeer"
echo "========"

echo "GCN"
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=0
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=2
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=4
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=6
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=8
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=10
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=15
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=20
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=25
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=30
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=35
python3 -m webkb.gcn --dataset=Chameleon --runs=1 --removal_nodes=40

