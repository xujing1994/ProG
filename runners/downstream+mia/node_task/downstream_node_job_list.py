import os
D_N_Ts = ["CiteSeer", "Cora", "PubMed", "Actor", "Wisconsin", "Texas", "ogbn-arxiv"]
idx_dnt = range(len(D_N_Ts))

D_G_Ts = ["MUTAG", "IMDB-BINARY", "COLLAB", "PROTEINS", "ENZYMES", "DD", "COX2", "BZR"]

GNNs = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'GCov', 'GraphTransformer']
idx_gnn = range(len(GNNs))

Pretrains = ['GraphCL', 'SimGRACE']
idx_pretrain = range(len(Pretrains))

Prompts = ['All-in-one', 'GPF', 'GPF-plus']
idx_prompt = range(len(Prompts))

Seeds = range(5)
idx_seed = range(len(Seeds))

N_Ss = range(1, 11)
idx_ns = range(len(N_Ss))

template_path = "./downstream_node_template.sh"
script_original = "python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/DATASET/PRETRAIN.GNNTYPE.128hidden_dim.pth --task NodeTask --dataset_name DATASET --gnn_type GNNTYPE --prompt_type PROMPTTYPE --shot_num SHOTNUM --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed SEED --epochs 300"

def main():
    script = "python3 downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/DATASET/PRETRAIN.GNNTYPE.128hidden_dim.pth --task NodeTask --dataset_name DATASET --gnn_type GNNTYPE --prompt_type PROMPTTYPE --shot_num SHOTNUM --hid_dim 128 --num_layer 2 --lr 1e-3 --decay 2e-6 --seed SEED --epochs 300"
    count = 0
    for i_dnt, D_N_T in zip(idx_dnt[4:5], D_N_Ts[4:5]):
        template = open(template_path, "r").read()
        template = template.replace('i_dataset', "D"+str(i_dnt))
        sbatch_path = "./downstream_node_{}.sh".format(D_N_T)
        with open(sbatch_path, "w") as f:
            f = f.write(template)
        for i_gnn, GNN in zip(idx_gnn, GNNs):
            for i_pretrain, Pretrain in zip(idx_pretrain, Pretrains):
                for i_prompt, Prompt in zip(idx_prompt, Prompts[:]):
                    for i_seed, Seed in zip(idx_seed[:1], Seeds[:1]):
                        for i_ns, N_S in zip(idx_ns[-1:], N_Ss[-1:]):
                            script = script.replace('PRETRAIN', str(Pretrain))
                            script = script.replace('DATASET', str(D_N_T))
                            script = script.replace('SEED', str(Seed))
                            script = script.replace('GNNTYPE', str(GNN))
                            script = script.replace("PROMPTTYPE", str(Prompt))
                            script = script.replace('SHOTNUM', str(N_S))
                            with open(sbatch_path, "a") as f:
                                f.write(script)
                                f.write('\n')
                            script = script_original
                            count += 1
        with open(sbatch_path, "a") as f:
            f.write("mv /home/c01jixu/CISPA-scratch/c01jixu/job-$SLURM_JOB_ID.out $JOBDATADIR/out.txt")
        print(count)


if __name__ == "__main__":
    main()