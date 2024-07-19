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

def read_data(path, shot_num):
    d = np.zeros(((len(Seeds), 70, 8))) # [len(Pretrains), len(Prompts), 7]
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

def draw_bar(outs_train_100, outs_test_100, pre_train_type, prompt_type, dataset, pre_train_data, model, shot_num):
    n_bins = 10
    labels_train_100 = outs_train_100[:, :, -1].astype(np.int64)
    labels_test_100 = outs_test_100[:, :, -1].astype(np.int64)
    prob_train_100 = outs_train_100[:, :, :-1]
    prob_test_100 = outs_test_100[:, :, :-1]
    # prob_train = outs_train[torch.arange(outs_train.size(0)), labels_train].numpy()
    # prob_test = outs_test[torch.arange(outs_test.size(0)), labels_test].numpy()
    prob_train = np.zeros((len(Seeds), 70))
    prob_test = np.zeros((len(Seeds), 70))
    for i in range(len(Seeds)):
        prob_train[i] = prob_train_100[i][np.arange(prob_train_100.shape[1]), labels_train_100[i]]
        prob_test[i] = prob_test_100[i][np.arange(prob_test_100.shape[1]), labels_test_100[i]]

    x = [prob_train.flatten(), prob_test.flatten()]

    fig, ax = plt.subplots(layout='constrained')
    colors = ['blue', 'red']
    labels = ['Member', 'Non member']
    ax.hist(x, n_bins, density=True, histtype='bar', color=colors, label=labels)
    # plt.show()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Density')
    ax.set_xlabel('Target prediction probability')
    ax.set_title('Probability distribution ({}_{}_{}_{}+{}+{}shot_100)'.format(pre_train_type, prompt_type, pre_train_data, dataset, model, shot_num))
    ax.legend(loc='upper left')

    # plt.show()
    save_path = "./figs/outs/" + '{}_{}_{}_{}_{}_{}_outs_100.png'.format(pre_train_type, prompt_type, pre_train_data, dataset, model, shot_num)
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1200)
    plt.close()
    print('save fig done')

if __name__ == "__main__":
    dataset = 'Cora'
    pre_train_data = 'PubMed'
    use_different_dataset = True
    shot_num = 10
    pre_train_type = ['GraphCL', 'SimGRACE', 'GraphMAE', 'DGI', 'Edgepred_GPPT', 'Edgepred_Gprompt']
    prompt_type = 'All-in-one'
    model='GCN'
    if use_different_dataset:
        path = "./Experiment_diff_dataset/outs/Node/{}shot/{}_{}".format(shot_num, dataset, pre_train_data)
    else:
        path = "./Experiment/outs/Node/{}shot/{}_{}".format(shot_num, dataset, pre_train_data)
    for ptt in pre_train_type[1:2]:
        outs_train_path = os.path.join(path, "train/{}_{}_{}.txt".format(ptt, prompt_type, model))
        outs_train_100 = read_data(outs_train_path, shot_num)

        outs_test_path = os.path.join(path, "test/{}_{}_{}.txt".format(ptt, prompt_type, model))
        outs_test_100 = read_data(outs_test_path, shot_num)

    # outs_train = torch.load(os.path.join(path, "train/{}_{}_{}.txt".format(pre_train_type, prompt_type, model)))
    # outs_test = torch.load(os.path.join(path, "{}_{}_{}_outs_test.pt".format(pre_train_type, prompt_type, model)), map_location='cpu')
    # labels_train = torch.load(os.path.join(path, "{}_{}_{}_labels_train.pt".format(pre_train_type, prompt_type, model)), map_location='cpu')
    # labels_test = torch.load(os.path.join(path, "{}_{}_{}_labels_test.pt".format(pre_train_type, prompt_type, model)), map_location='cpu')
        draw_bar(outs_train_100, outs_test_100, ptt, prompt_type, dataset=dataset, pre_train_data = pre_train_data, model=model, shot_num=shot_num)
    # print the maximum result for each prompt method among six pre-training methods

def draw_figure(data_avg_adv, data_std_adv, data_avg_ben, data_std_ben, dataset, recover_from):
    filename = '{}_{}.pdf'.format(dataset, model)
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 5))
    color_list = ['tab:green', 'tab:blue', 'tab:orange', 'tab:brown', 'tab:purple', 'tab:red']
    label_list = ['GAT', 'GIN', 'GraphSAGE']
    x = x_dict[dataset]
    titles = ['Surrogate Accuracy', 'Surrogate Fidelity']
    #fig.suptitle('{}_{}'.format(dataset, recover_from), fontsize=16)
    for i in range(len(titles)):
        for i_m, model in enumerate(models):
            axs[i].plot(x, data_avg_adv[i_m, :, 2+i], color=color_list[i_m], linewidth=2.0, linestyle='-', marker='v', markersize=10)
            axs[i].plot(x, data_avg_ben[i_m, :, 2+i], color=color_list[i_m], linewidth=2.0, linestyle='-', marker='s', markersize=10)
            # axs[i].plot(x, data_avg_cdn[i_m, :, 2+i], color=color_list[i_m], linewidth=1.0, linestyle='--', marker='*', markersize=5)

            axs[i].fill_between(x, (data_avg_adv[i_m, :, 2+i]-data_std_adv[i_m, :, 2+i]), (data_avg_adv[i_m, :, 2+i]+data_std_adv[i_m, :, 2+i]), \
                color=color_list[i_m], alpha=0.08)
            axs[i].fill_between(x, (data_avg_ben[i_m, :, 2+i]-data_std_ben[i_m, :, 2+i]), (data_avg_ben[i_m, :, 2+i]+data_std_ben[i_m, :, 2+i]), \
                color=color_list[i_m], alpha=0.08)
            # axs[i].fill_between(x, (data_avg_cdn[i_m, :, 2+i]-data_std_cdn[i_m, :, 2+i]), (data_avg_cdn[i_m, :, 2+i]+data_std_cdn[i_m, :, 2+i]), \
            #     color=color_list[i_m], alpha=0.08)
            # draw target model testing accuracy as the baseline
            # if i == 0:
            #     axs[i].plot(x, data_avg_adv[i_m, :, 1], color=color_list[i_m], linewidth=1.0, linestyle='dashdot')
    save_dir = './figs/attack_performance_new/'
    for i_a, ax in enumerate(axs):
        ax.set(xlabel='Query Rate')
        ax.set(xticks=x_dict[dataset])
        #if dataset == 'amazon_photo':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid()
    for ax in axs:
        ax.set(yticks=y_label_dict[dataset][0], ylim=y_label_dict[dataset][1])
    handles = axs[0].get_legend_handles_labels()
    axs[0].set(ylabel=titles[0])
    axs[1].set(ylabel=titles[1])
    gat_line = mlines.Line2D([], [], color=color_list[0], label='GAT', linestyle='-', linewidth=2.0)
    gin_line = mlines.Line2D([], [], color=color_list[1], label='GIN', linestyle='-', linewidth=2.0)
    sage_line = mlines.Line2D([], [], color=color_list[2], label='SAGE', linestyle='-', linewidth=2.0)
    adv_line = mlines.Line2D([], [], color='tab:gray', marker='v',
                          markersize=10, label='Strategy 1', linestyle='-', linewidth=2.0)
    # cdn_line = mlines.Line2D([], [], color='tab:gray', marker='*',
    #                       markersize=5, label='adv_cdn', linestyle='--', linewidth=1.0)
    ben_line = mlines.Line2D([], [], color='tab:gray', label='Strategy 2', marker='s', markersize=10, linestyle='-', linewidth=2.0)  
    # target_model_line =  mlines.Line2D([], [], color='tab:gray', label='TM acc', linestyle='dashdot', linewidth=1.0)  
    handles[0].append(gat_line)
    handles[0].append(gin_line)
    handles[0].append(sage_line)
    handles[0].append(adv_line)
    handles[0].append(ben_line)  
    # handles[0].append(target_model_line)    
    # handles[0].append(cdn_line)         
    #if recover_from == 'embedding':  
    #plt.legend(bbox_to_anchor=(-1.3,-0.15), loc="upper left", handles=handles[0], ncol=len(handles[0]))
    legend = plt.legend(loc="lower right", fancybox=True, handles=handles[0], prop={'size': 13}, ncol=len(handles[0]), bbox_to_anchor=(-1.3,-0.15))

    # save legend figure
    ipdb.set_trace()
    legend_path = save_dir + '{}_{}_legend.pdf'.format(dataset, recover_from)
    export_legend(legend, legend_path)

    save_path = save_dir + filename
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    #plt.show()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print('save fig done')
