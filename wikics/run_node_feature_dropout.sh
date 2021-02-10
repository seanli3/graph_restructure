#!/bin/sh

#echo "Cora"
#echo "===="
#
#echo "GCN"
#python gcn.py --dataset=Cora --node_feature_dropout=1
#
#echo "GAT"
#python gat.py --dataset=Cora --node_feature_dropout=1
#
#echo "Cheby"
#python cheb.py --dataset=Cora --num_hops=3 --node_feature_dropout=1
#
#echo "SGC"
#python sgc.py --dataset=Cora --K=3 --weight_decay=0.0005 --node_feature_dropout=1
#
#echo "ARMA"
#python arma.py --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights=True --node_feature_dropout=1
#
#echo "APPNP"
#python appnp.py --dataset=Cora --alpha=0.1 --node_feature_dropout=1


echo "CiteSeer"
echo "========"

#echo "GCN"
#python gcn.py --dataset=CiteSeer --node_feature_dropout=1
#
#echo "GAT"
#python gat.py --dataset=CiteSeer --node_feature_dropout=1
#
#echo "Cheby"
#python cheb.py --dataset=CiteSeer --num_hops=2 --node_feature_dropout=1
#
#echo "SGC"
#python sgc.py --dataset=CiteSeer --K=2 --weight_decay=0.005 --node_feature_dropout=1
#
#echo "ARMA"
#python arma.py --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights=True --node_feature_dropout=1
#
#echo "APPNP"
#python appnp.py --dataset=CiteSeer --alpha=0.1 --node_feature_dropout=1

echo "Our model"
#python decimation.py --node_feature_dropout=1 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
python decimation.py --node_feature_dropout=0.9 --dataset=Citeseer --alpha=0.2 --lr=0.0005 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=8000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.8 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.7 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.6 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.5 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.4 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.3 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.2 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001
#python decimation.py --node_feature_dropout=0.1 --dataset=Citeseer --alpha=0.2 --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --cuda --order=16 --early_stopping=10 --epochs=10000 --weight_decay=0.001

#echo "PubMed"
#echo "======"
#
#echo "GCN"
#python gcn.py --dataset=PubMed --node_feature_dropout=1
#
#echo "GAT"
#python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --node_feature_dropout=1
#
#echo "Cheby"
#python cheb.py --dataset=PubMed --num_hops=2 --node_feature_dropout=1
#
#echo "SGC"
#python sgc.py --dataset=PubMed --K=2 --weight_decay=0.0005 --node_feature_dropout=1
#
#echo "ARMA"
#python arma.py --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.1 --node_feature_dropout=1
#
#echo "APPNP"
#python appnp.py --dataset=PubMed --alpha=0.1 --node_feature_dropout=1

read junk
