import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ipdb
import statistics
import copy
import matplotlib
import random
import pickle
import torch
import ipdb
import torchmetrics
from sklearn.manifold import TSNE

# matplotlib.rc('xtick', labelsize=25) 
# matplotlib.rc('ytick', labelsize=25) 
# matplotlib.rc('font', size=25) 

D_N_Ts = ["CiteSeer", "Cora", "PubMed", "Actor", "Wisconsin", "Texas", "ogbn-arxiv"]
idx_dnt = range(len(D_N_Ts))

D_G_Ts = ["MUTAG", "IMDB-BINARY", "COLLAB", "PROTEINS", "ENZYMES", "DD", "COX2", "BZR"]

GNNs = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'GCov', 'GraphTransformer']
idx_gnn = range(len(GNNs))

Pretrains = ['GraphCL', 'SimGRACE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'DGI', 'GraphMAE']

Prompts = ['All-in-one', 'GPF', 'GPF-plus', 'GPPT']

Seeds = range(100)

N_Ss = range(1, 11)

accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=7).to('cpu')

def read_data(path, shot_num):
    d = np.zeros(((len(Seeds), int(shot_num*7), 8))) # [len(Pretrains), len(Prompts), 7]
    count = np.zeros(len(Seeds))
    with open(path, 'r') as f:
        for line in f.readlines()[:]:
            # if line.split(' ')[0] == 'pre_train+prompt':
            #     continue
            # line = line.replace('+0.0', '')
            line = line.replace('\n', '')
            data = line.split(' ')
            # pretrain_type = data[0].split('+')[0]
            # prompt_type = data[0].split('+')[1]
            seed = int(data[0])
            d[seed, int(count[seed])] = data[1:]
            count[seed] += 1
    # # check whether all results are ready:
    #     for i in range(len(Pretrains)):
    #         if d[i, 0, 0] == 0:
    #             print(Pretrains[i], path)
    return d

def draw_tsne(features_train, features_test, labels_train, labels_test, pre_train_type, prompt_type, dataset, pre_train_data, model, shot_num, logit_scaling=False):
    # ipdb.set_trace()
    features_all = np.concatenate((features_train.numpy(), features_test.numpy()), axis=0) # visualize the features of the target label --> but the number of features is not the same as that of the labels
    tsne = TSNE(n_components=3, learning_rate='auto', perplexity=10, verbose=True)
    features_transformed = tsne.fit_transform(features_all)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    train_idx = range(len(features_all))[:int(len(features_all)/2)]
    test_idx = range(len(features_all))[int(len(features_all)/2):]
    ax.scatter(features_transformed[train_idx, 0], features_transformed[train_idx, 1], features_transformed[train_idx, 2], label='member', alpha=0.5)
    ax.scatter(features_transformed[test_idx, 0], features_transformed[test_idx, 1], features_transformed[test_idx, 2], label='non-member', alpha=0.5)

    colors = ['blue', 'red']
    labels = ['Member', 'Non member']
    # plt.show()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('')
    if logit_scaling:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('')
    ax.set_title('Train & Test Distribution ({}_{}_{}_{}+{}+{}shot_100)'.format(pre_train_type, prompt_type, pre_train_data, dataset, model, shot_num))
    ax.legend(loc='upper left')
    # ax.text(-10, 0.02, 'train acc: {:.4f} \n test acc: {:.4f}'.format(acc_train, acc_test)) # position for outs: (0.1, 5.0); for logit_scaling: (-10, 0.02)

    # plt.show()
    if logit_scaling:
        save_path = "./figs/prompted_features/{}shot/".format(shot_num) + '{}_{}_{}_{}_{}.png'.format(pre_train_type, prompt_type, pre_train_data, dataset, model)
    else:
        save_path = "./figs/prompted_features/{}shot/".format(shot_num) + '{}_{}_{}_{}_{}.png'.format(pre_train_type, prompt_type, pre_train_data, dataset, model)
    print(save_path)
    isExist = os.path.exists(os.path.split(save_path)[0])
    if not isExist:
        os.makedirs(os.path.split(save_path)[0])
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1200)
    plt.close()
    print('save fig done')

if __name__ == "__main__":
    dataset = 'Cora'
    pre_train_data = ['CiteSeer','PubMed', 'Photo', 'ogbn-arxiv']
    use_different_dataset = True
    shot_nums = range(1, 6)
    pre_train_type = ['GraphCL', 'SimGRACE', 'GraphMAE', 'DGI', 'Edgepred_GPPT', 'Edgepred_Gprompt']
    prompt_type = 'GPF-plus'
    model='GCN'
    for shot_num in shot_nums[:1]:
        for ptd in pre_train_data[:1]:
            if use_different_dataset:
                path = "./Experiment_diff_dataset/prompted_features/Node/{}shot/{}_{}".format(shot_num, dataset, ptd)
            else:
                path = "./Experiment/prompted_features/Node/{}shot/{}_{}".format(shot_num, dataset, ptd)
            for ptt in pre_train_type[2:3]:
                features_train = torch.load(os.path.join(path, 'train/{}_{}_{}_features.pt'.format(ptt, prompt_type, model)), map_location='cpu')
                features_test = torch.load(os.path.join(path, 'test/{}_{}_{}_features.pt'.format(ptt, prompt_type, model)), map_location='cpu')
                labels_train = torch.load(os.path.join(path, 'train/{}_{}_{}_labels.pt'.format(ptt, prompt_type, model)), map_location='cpu')
                labels_test = torch.load(os.path.join(path, 'test/{}_{}_{}_labels.pt'.format(ptt, prompt_type, model)), map_location='cpu')
                draw_tsne(features_train, features_test, labels_train, labels_test, ptt, prompt_type, dataset=dataset, pre_train_data = ptd, model=model, shot_num=shot_num)
