export LD_LIBRARY_PATH=/apps/cuda/10.1/lib64:$LD_LIBRARY_PATH
export CPATH=/apps/cuda/10.1/include:$CPATH

python3 ./citation/decimation.py $LINE
