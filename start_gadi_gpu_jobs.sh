echo "Reading parameters from file: " $1
while read -r line; do qsub -v LINE="$line" ./run_gadi_gpu.sh; done < "$1"
