#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --output=/home/c01jixu/CISPA-scratch/c01jixu/job-%j.out

#SBATCH --gres=gpu:A100:1

if [ ! -f ~/.config/enroot/.credentials ]; then
        mkdir -p ~/.config/enroot/
        ln -s ~/CISPA-home/.config/enroot/.credentials ~/.config/enroot/.credentials
fi

JOBDATADIR=/home/"$USER"/CISPA-home/"$USER"/job-"$SLURM_JOB_ID"/

srun mkdir -p $JOBDATADIR
# srun mkdir -p "$JOBDATADIR"/models

PROJDIR=/home/c01jixu/CISPA-home/ProG
pwd
srun --container-image=projects.cispa.saarland:5005#c01jixu/prog:24.03-py3 --container-mounts="$JOBDATADIR":/tmp python3 "$PROJDIR"/pre_train.py --task GraphCL --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 10 --seed 1 --lr 1e-3 --decay 2e-6

srun mv /home/c01jixu/CISPA-scratch/c01jixu/job-$SLURM_JOB_ID.out "$JOBDATADIR"/out.txt
srun mv "$JOBDATADIR"/job-"$SLURM_JOB_ID"/logs "$JOBDATADIR"/logs
srun rm -rf "$JOBDATADIR"/job-"$SLURM_JOB_ID"