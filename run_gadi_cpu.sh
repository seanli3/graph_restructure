#!/bin/bash
#PBS -P go95
#PBS -q express
#PBS -l walltime=10:00:00
#PBS -l mem=2GB
#PBS -l jobfs=200MB
#PBS -l ncpus=2
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=train_old_splot
## The job will be executed from current working directory instead of home.
#PBS -l wd

LINE=$LINE ./gadi_decimation_citation.sh
