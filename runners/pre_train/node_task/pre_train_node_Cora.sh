#!/bin/bash

#SBATCH --container-image projects.cispa.saarland:5005\#c01jixu/prog:24.03-py3
#SBATCH --container-mounts=/home/c01jixu/CISPA-home/tmp:/tmp
#SBATCH --container-workdir=/home/c01jixu/CISPA-home/ProG
#SBATCH --gres=gpu:A100:1
#SBATCH --partition tmp
#SBATCH --output=/home/c01jixu/CISPA-scratch/c01jixu/job-%j.out
#SBATCH --time=2:00:00
#SBATCH --job-name=D1

pwd

JOBDATADIR=/home/"$USER"/CISPA-home/"$USER"/job-"$SLURM_JOB_ID"/

mkdir -p $JOBDATADIR
python3 pre_train.py --task 'GraphCL' --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'SimGRACE' --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'GraphCL' --dataset_name 'Cora' --gnn_type 'GAT' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'SimGRACE' --dataset_name 'Cora' --gnn_type 'GAT' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'GraphCL' --dataset_name 'Cora' --gnn_type 'GIN' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'SimGRACE' --dataset_name 'Cora' --gnn_type 'GIN' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'GraphCL' --dataset_name 'Cora' --gnn_type 'GraphSAGE' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'SimGRACE' --dataset_name 'Cora' --gnn_type 'GraphSAGE' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'GraphCL' --dataset_name 'Cora' --gnn_type 'GCov' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'SimGRACE' --dataset_name 'Cora' --gnn_type 'GCov' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'GraphCL' --dataset_name 'Cora' --gnn_type 'GraphTransformer' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
python3 pre_train.py --task 'SimGRACE' --dataset_name 'Cora' --gnn_type 'GraphTransformer' --hid_dim 128 --num_layer 2 --epochs 300 --seed 0 --lr 1e-3 --decay 2e-6
mv /home/c01jixu/CISPA-scratch/c01jixu/job-$SLURM_JOB_ID.out $JOBDATADIR/out.txt