import numpy as np

parameters = []
# hidden=88 heads = 16 dropout=0.75 cheby_order=13 alpha=0.2
for stop in [50]: 
    parameters.append('--dataset=Citeseer --alpha=0.2 --lr=0.005 --hidden=88 --heads=16 --dropout=0.75 --cuda --order=13 --early_stopping=50 --epochs=400 --weight_decay=0.0005 --filter=analysis --runs=10\n'.format(str(stop)))

with open('citeseer_parameters.txt', 'w') as the_file:
    the_file.writelines(parameters)

