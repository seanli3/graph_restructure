#!/bin/sh


python -m citation.decimation --dataset=Cora --Kb=18 --Ka=12 --dropout=0.1 --heads=6 --hidden=32 --lr=0.04 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005 --method=ARMA --Tmax=20 --k=18 --self_loop
python -m citation.decimation --dataset=Cora --Kb=18 --Ka=12 --dropout=0.1 --heads=6 --hidden=64 --lr=0.04 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005 --method=ARMA --Tmax=20 --k=18 --self_loop
python -m citation.decimation --dataset=Cora --Kb=18 --Ka=12 --dropout=0.1 --heads=6 --hidden=128 --lr=0.04 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005 --method=ARMA --Tmax=20 --k=18 --self_loop
python -m citation.decimation --dataset=Cora --Kb=18 --Ka=12 --dropout=0.1 --heads=6 --hidden=256 --lr=0.04 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005 --method=ARMA --Tmax=20 --k=18 --self_loop
python -m citation.decimation --dataset=Cora --Kb=18 --Ka=12 --dropout=0.1 --heads=6 --hidden=512 --lr=0.04 --patience=100 --seed=729 --epochs=2000 --runs=1 --weight_decay=0.0005 --method=ARMA --Tmax=20 --k=18 --self_loop

