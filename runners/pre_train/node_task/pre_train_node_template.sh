#!/bin/bash

#SBATCH --container-image projects.cispa.saarland:5005\#c01jixu/prog:24.03-py3
#SBATCH --container-mounts=/home/c01jixu/CISPA-home/tmp:/tmp
#SBATCH --container-workdir=/home/c01jixu/CISPA-home/ProG
#SBATCH --gres=gpu:A100:1
#SBATCH --partition tmp
#SBATCH --output=/home/c01jixu/CISPA-home/ProG/runners/outputs/job-%j.out
#SBATCH --time=20:00:00
#SBATCH --job-name=i_dataset

pwd
