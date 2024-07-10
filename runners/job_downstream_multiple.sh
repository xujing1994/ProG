#!/bin/bash

#SBATCH --container-image projects.cispa.saarland:5005\#c01jixu/prog:24.03-py3
#SBATCH --container-mounts=/home/c01jixu/CISPA-home/tmp:/tmp
#SBATCH --container-workdir=/home/c01jixu/CISPA-home/ProG
#SBATCH --gres=gpu:A100:1
#SBATCH --partition tmp
#SBATCH --output=/home/c01jixu/CISPA-scratch/c01jixu/job-%j.out
#SBATCH --time=02:00:00


JOBDATADIR=/home/"$USER"/CISPA-home/"$USER"/job-"$SLURM_JOB_ID"/

mkdir -p $JOBDATADIR

python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/CiteSeer/GraphCL.GCN.128hidden_dim.pth --task NodeTask --dataset_name 'CiteSeer' --gnn_type 'GCN' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed 1 --epochs 300

python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/CiteSeer/GraphCL.GAT.128hidden_dim.pth --task NodeTask --dataset_name 'CiteSeer' --gnn_type 'GAT' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed 1 --epochs 300

python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/CiteSeer/GraphCL.GraphSAGE.128hidden_dim.pth --task NodeTask --dataset_name 'CiteSeer' --gnn_type 'GraphSAGE' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed 1 --epochs 300

python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/CiteSeer/GraphCL.GCN.128hidden_dim.pth --task NodeTask --dataset_name 'CiteSeer' --gnn_type 'GraphTransformer' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed 1 --epochs 300

python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/CiteSeer/GraphCL.GAT.128hidden_dim.pth --task NodeTask --dataset_name 'CiteSeer' --gnn_type 'GIN' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed 1 --epochs 300

python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/CiteSeer/GraphCL.GraphSAGE.128hidden_dim.pth --task NodeTask --dataset_name 'CiteSeer' --gnn_type 'GCov' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed 1 --epochs 300


mv /home/c01jixu/CISPA-scratch/c01jixu/job-$SLURM_JOB_ID.out "$JOBDATADIR"/out.txt
mv "$JOBDATADIR"/job-"$SLURM_JOB_ID"/logs "$JOBDATADIR"/logs
rm -rf "$JOBDATADIR"/job-"$SLURM_JOB_ID"