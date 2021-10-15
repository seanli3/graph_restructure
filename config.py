from pathlib import Path
import os

path = Path(__file__).parent

SAVED_MODEL_DIR_NODE_CLASSIFICATION = path / 'models/saved_models/node_classification/'
SAVED_MODEL_DIR_GRAPH_CLASSIFICATION = path / 'models/saved_models/graph_classification/'

for p in [SAVED_MODEL_DIR_GRAPH_CLASSIFICATION, SAVED_MODEL_DIR_GRAPH_CLASSIFICATION]:
    if not os.path.exists(p):
        os.makedirs(p)

SAVED_MODEL_PATH_NODE_CLASSIFICATION = str(SAVED_MODEL_DIR_NODE_CLASSIFICATION / '{}_best_model_split_{}.pt')
TU_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION = str(SAVED_MODEL_DIR_NODE_CLASSIFICATION / '{}_dataset.pt')
OGB_SAVED_MODEL_PATH_GRAPH_CLASSIFICATION = str(SAVED_MODEL_DIR_NODE_CLASSIFICATION / '{}_dataset_split_{}.pt')
