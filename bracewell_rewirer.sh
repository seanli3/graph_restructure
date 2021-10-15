source ~/.bashrc
export LD_LIBRARY_PATH=/apps/cuda/11.1.1/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/11.1.1/include:$CPATH

module load cuda/11.1.1
module load miniconda3/4.9.2 
conda activate /datastore/li243/.conda/env/rewire/

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 -m graph_dictionary.spectral_rewire_dictionary $LINE
