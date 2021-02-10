#!/bin/sh

echo "Cora"
echo "===="

echo "hidden=8"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=8 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=16"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=16 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=24"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=24 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=32"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=32 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=40"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=40 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=48"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=48 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=56"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=56 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
echo "hidden=64"
echo "===="
python decimation.py --dataset=Cora --lr=0.0001 --hidden=64 --heads=8 --dropout=0.95 --order=16 --weight_decay=0.001
