import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ipdb
import statistics
import copy
import matplotlib
import random

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

Seeds = range(5)

N_Ss = range(1, 11)

def read_data(path):
    d = np.zeros(((len(Pretrains), len(Prompts), 7))) # [len(Pretrains), len(Prompts), 7]
    with open(path, 'r') as f:
        for line in f.readlines()[:]:
            if line.split(' ')[0] == 'pre_train+prompt':
                continue
            line = line.replace('+0.0', '')
            line = line.replace('\n', '')
            data = line.split(' ')
            pretrain_type = data[0].split('+')[0]
            prompt_type = data[0].split('+')[1]

            for i, pre in enumerate(Pretrains):
                for j, pt in enumerate(Prompts):
                    if pretrain_type == pre and prompt_type == pt:
                        #ipdb.set_trace()
                        d[i, j] = data[8:]
    # check whether all results are ready:
        for i in range(len(Pretrains)):
            if d[i, 0, 0] == 0:
                print(Pretrains[i], path)
    return d

def draw_bar(data, dataset, pre_train_data, model, shot_num):
    species = ('GraphCL', 'SimGRACE', 'Edgepred_ \n GPPT', 'Edgepred_ \n Gprompt', 'DGI', 'GraphMAE')
    # penguin_means = {
    #     'All-in-one': data[:, 0, 0],
    #     'GPF': data[:, 1, 0],
    #     'Flipper Length': (189.95, 195.82, 217.19),
    # }

    x = np.arange(len(species))  # the label locations
    width = 0.16  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    # for attribute, measurement in zip(Prompts, data[:, :, 0]):
    for i, pt in enumerate(Prompts):
        attribute = pt
        measurement = data[:, i, 0]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Downstream task ({}_{}+GCN+{}shot)'.format(pre_train_data, dataset, shot_num))
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    # plt.show()
    save_path = "./figs/" + '{}_{}_{}_{}_new.png'.format(pre_train_data, dataset, model, shot_num)
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1200)
    plt.close()
    print('save fig done')

if __name__ == "__main__":
    dataset = 'Cora'
    pre_train_data = 'Cora'
    use_different_dataset = True
    shot_num = 100
    if use_different_dataset:
        path = "./Experiment_diff_dataset/ExcelResults/Node/{}shot/{}_{}/GCN_total_results.txt".format(shot_num, dataset, pre_train_data)
    else:
        path = "./Experiment/ExcelResults/Node/{}shot/{}_{}/GCN_total_results.txt".format(shot_num, dataset, pre_train_data)
    
    data = read_data(path)
    print(data)
    draw_bar(data, dataset=dataset, pre_train_data = pre_train_data, model='GCN', shot_num=shot_num)
    # print the maximum result for each prompt method among six pre-training methods

    for i in range(len(Prompts)):
        print('{}: {:.4}%'.format(Prompts[i], max(data[:, i, 0]*100)))
        print(['{:.4}%'.format(k) for k in data[:, i, -1]*100])

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
