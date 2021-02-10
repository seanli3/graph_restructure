export LD_LIBRARY_PATH=/apps/cuda/10.1.168/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/10.1.168/include:$CPATH

module load python/3.7.2
python3 ./citation/decimation_param_tuning.py $LINE
