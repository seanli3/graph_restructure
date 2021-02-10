echo "Reading parameters from file: " $1
while read -r line; do qsub -v LINE="$line" ./run_gadi_cpu.sh; done < "$1"
