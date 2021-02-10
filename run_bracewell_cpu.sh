#!/bin/bash
#SBATCH --job-name=tuning_pubmed
#SBATCH --time=128:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --output=%x-%j.out

LINE=$LINE ./bracewell_decimation_citation.sh
