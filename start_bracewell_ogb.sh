echo "Reading parameters from file: " $1
while read -r line; do sbatch --export LINE="$line" run_bracewell_ogb.sh; done < "$1"
