import os
D_N_Ts = ["CiteSeer", "Cora", "PubMed", "Actor", "Wisconsin", "Texas", "ogbn-arxiv"]
idx_dnt = range(len(D_N_Ts))

D_G_Ts = ["MUTAG", "IMDB-BINARY", "COLLAB", "PROTEINS", "ENZYMES", "DD", "COX2", "BZR"]

GNNs = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'GCov', 'GraphTransformer']
idx_gnn = range(len(GNNs))

Pretrains = ['GraphCL', 'SimGRACE']

Prompts = ['All-in-one', 'GPF', 'GPF-plus']

Seeds = range(5)

N_Ss = range(1, 11)

os.system("pwd")
template_path = "./pre_train_node_template.sh"
script_original = "python3 pre_train.py --task 'PRETRAIN' --dataset_name 'DATASET' --gnn_type 'GNN' --hid_dim 128 --num_layer 2 --epochs 300 --seed SEED --lr 1e-3 --decay 2e-6"

def main():
    script = "python3 pre_train.py --task 'PRETRAIN' --dataset_name 'DATASET' --gnn_type 'GNN' --hid_dim 128 --num_layer 2 --epochs 300 --seed SEED --lr 1e-3 --decay 2e-6"
    for i_dnt, D_N_T in zip(idx_dnt[5:6], D_N_Ts[5:6]):
        count = 0
        template = open(template_path, "r").read()
        template = template.replace('i_dataset', "D"+str(i_dnt))
        sbatch_path = "./pre_train_node_{}.sh".format(D_N_T)
        with open(sbatch_path, "w") as f:
            f = f.write(template)
        for i_gnn, GNN in enumerate(GNNs):
            for i_pretrain, Pretrain in enumerate(Pretrains):
                for i_prompt, Prompt in enumerate(Prompts[:1]):
                    for i_seed, Seed in enumerate(Seeds[:1]):
                        for i_ns, N_S in enumerate(N_Ss[:1]):
                            script = script.replace('PRETRAIN', str(Pretrain))
                            script = script.replace('DATASET', str(D_N_T))
                            script = script.replace('SEED', str(Seed))
                            script = script.replace('GNN', str(GNN))
                            with open(sbatch_path, "a") as f:
                                f.write(script)
                                f.write('\n')
                            script = script_original
                            count += 1
        # with open(sbatch_path, "a") as f:
        #     f.write("mv /home/c01jixu/CISPA-scratch/c01jixu/job-$SLURM_JOB_ID.out $JOBDATADIR/out.txt")
        #os.system('sbatch {}'.format(sbatch_path))
    print(count)

if __name__ == "__main__":
    main()