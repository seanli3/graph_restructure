from pathlib import Path
import torch
import os

path = Path(__file__).parent

SAVED_MODEL_DIR_NODE_CLASSIFICATION = str(path / 'models/saved_models/node_classification')
SAVED_MODEL_DIR_GRAPH_CLASSIFICATION = str(path / 'models/saved_models/graph_classification')

for p in [SAVED_MODEL_DIR_GRAPH_CLASSIFICATION, SAVED_MODEL_DIR_GRAPH_CLASSIFICATION]:
    if not os.path.exists(p):
        os.makedirs(p)

TU_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '{}_dataset.pt'
OGB_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION = SAVED_MODEL_DIR_NODE_CLASSIFICATION + '{}_dataset_split_{}.pt'
USE_CUDA = False if os.getenv('CPU_ONLY') is not None else True
EDGE_LOGIT_THRESHOLD = 0.5
SEED=729

DEVICE = 'cpu'
if USE_CUDA and torch.cuda.is_available():
    import torch
    device = os.getenv('CUDA_DEVICE')
    if device is None:
        device = '0'
    DEVICE = torch.device('cuda:'+device)
    print('Using device:' + str(DEVICE))

