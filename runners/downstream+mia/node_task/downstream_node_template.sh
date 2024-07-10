#!/bin/bash

#SBATCH --container-image projects.cispa.saarland:5005\#c01jixu/prog:24.03-py3
#SBATCH --container-mounts=/home/c01jixu/CISPA-home/tmp:/tmp
#SBATCH --container-workdir=/home/c01jixu/CISPA-home/ProG
#SBATCH --gres=gpu:A100:1
#SBATCH --partition tmp
#SBATCH --output=/home/c01jixu/CISPA-scratch/c01jixu/job-%j.out
#SBATCH --time=20:00:00
#SBATCH --job-name=Downstream_i_dataset

pwd

JOBDATADIR=/home/"$USER"/CISPA-home/"$USER"/job-"$SLURM_JOB_ID"/

mkdir -p $JOBDATADIR
