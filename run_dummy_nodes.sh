#!/bin/sh

echo "CiteSeer"
echo "========"

echo "GAT"
#python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=0
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=1
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=2
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=3
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=4
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=5
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=6
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=8
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=10
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=12
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=14
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=16
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=18
python3 -m citation.gat --dataset=CiteSeer --runs=1 --split=full --dummy_nodes=20

