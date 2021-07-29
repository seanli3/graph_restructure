#!/bin/bash
#SBATCH --job-name=pubmed
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --mem=24GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

python3 -m graph_dictionary.spectral_rewire_dictionary --dataset=PubMed
