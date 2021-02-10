#!/bin/bash
#SBATCH --job-name=tuning_gpu
#SBATCH --time=128:00:00
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

LINE=$LINE ./bracewell_decimation_citation.sh
