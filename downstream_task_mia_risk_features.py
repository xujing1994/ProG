from prompt_graph.tasker import NodeTask, GraphTask, MIATask
from prompt_graph.utils import seed_everything
# from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import os
import pandas as pd
import ipdb
import torch

def load_induced_graph(dataset_name, data, device, use_different_dataset):
    if use_different_dataset:
        folder_path = './Experiment_diff_dataset/induced_graph/' + dataset_name
    else:
        folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


args = get_args()
# seed_everything(args.seed)
print('dataset_name', args.dataset_name)

def main():
    if args.use_different_dataset:
        args.pre_train_model_path = './Experiment_diff_dataset/pre_trained_model/{}/{}.{}.128hidden_dim.pth'.format(args.pre_train_data, args.pre_train_type, args.gnn_type)
    else:
        args.pre_train_model_path = './Experiment/pre_trained_model/{}/{}.{}.128hidden_dim.pth'.format(args.pre_train_data, args.pre_train_type, args.gnn_type)
    # pretrain_path = args.pre_train_model_path
    # pretrain_type = os.path.split(pretrain_path)[1].split('.')[0]
    # print("Dataset: {}, Pre-train Data: {}, GNN: {}, Pretrain: {}, Prompt: {}, ShotNum: {}, Seed: {}".format(args.dataset_name, args.pre_train_data, args.gnn_type, args.pre_train_type, args.prompt_type, args.shot_num, args.seed))
    # if args.dataset_name not in args.pre_train_model_path :
    #      # use different dataset for prompt fine-tuning
    #      args.use_different_dataset = True
    # else:
    #     args.use_different_dataset = False
    #     args.pre_train_data = args.dataset_name
    if args.task == 'NodeTask' or args.task == 'FineTuneNodeTask':
        data, input_dim, output_dim = load4node(args.dataset_name, args.use_different_dataset)   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
            graphs_list = load_induced_graph(args.dataset_name, data, device, args.use_different_dataset)  # the debugging on Gprompt method is still in process
        else:
            graphs_list = None 
    if args.task == 'GraphTask':
        input_dim, output_dim, dataset = load4graph(args.dataset_name)
    
    features_train_list = []
    features_test_list = []
    labels_train_list = []
    labels_test_list = []
    for seed in range(2):
        args.seed = seed
        seed_everything(args.seed)
        print("Dataset: {}, Pre-train Data: {}, GNN: {}, Pretrain: {}, Prompt: {}, ShotNum: {}, Seed: {}".format(args.dataset_name, args.pre_train_data, args.gnn_type, args.pre_train_type, args.prompt_type, args.shot_num, args.seed))

        mia_features, mia_labels = get_mia_features(data, input_dim, output_dim, graphs_list)
        _, _, features_train, features_test = mia_features # outs_train: [7*shot_num, 7], features_train: [7*shot_num, input_dim]
        labels_train, labels_test = mia_labels # labels_train: [7*shot_num]
        features_train_list.append(features_train)
        features_test_list.append(features_test)
        labels_train_list.append(labels_train)
        labels_test_list.append(labels_test)
    features_train = torch.cat(features_train_list, dim=0)
    features_test = torch.cat(features_test_list, dim=0)
    labels_train = torch.cat(labels_train_list, dim=0)
    labels_test = torch.cat(labels_test_list, dim=0)
    # save the features_train, features_test, labels_train and labels_test
    if args.use_different_dataset:
        train_folder = './Experiment_diff_dataset/prompted_features/Node/{}shot/{}_{}/train'.format(args.shot_num, args.dataset_name, args.pre_train_data)
        test_folder = './Experiment_diff_dataset/prompted_features/Node/{}shot/{}_{}/test'.format(args.shot_num, args.dataset_name, args.pre_train_data)
    else:
        train_folder = './Experiment/prompted_features/Node/{}shot/{}_{}/train'.format(args.shot_num, args.dataset_name, args.pre_train_data)
        test_folder = './Experiment/prompted_features/Node/{}shot/{}_{}/test'.format(args.shot_num, args.dataset_name, args.pre_train_data)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    torch.save(features_train, os.path.join(train_folder, '{}_{}_{}_features.pt'.format(args.pre_train_type, args.prompt_type, args.gnn_type)))
    torch.save(features_test,os.path.join(test_folder, '{}_{}_{}_features.pt'.format(args.pre_train_type, args.prompt_type, args.gnn_type)))
    torch.save(labels_train, os.path.join(train_folder, '{}_{}_{}_labels.pt'.format(args.pre_train_type, args.prompt_type, args.gnn_type)))
    torch.save(labels_test, os.path.join(test_folder, '{}_{}_{}_labels.pt'.format(args.pre_train_type, args.prompt_type, args.gnn_type)))

def get_mia_features(data, input_dim, output_dim, graphs_list):
    if args.task == 'NodeTask':
        tasker_target = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, seed = args.seed, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list, use_different_dataset=args.use_different_dataset, pre_train_data=args.pre_train_data)
        tasker_shadow = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, seed = args.seed, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list, use_different_dataset=args.use_different_dataset, pre_train_data=args.pre_train_data)
        pre_train_type = tasker_target.pre_train_type
    elif args.task == 'FineTuneNodeTask':
        args.prompt_type = 'None'
        tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, seed = args.seed, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list, use_different_dataset=args.use_different_dataset)
        pre_train_type = tasker.pre_train_type
   
    elif args.task == 'GraphTask':
        tasker_target = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim, pre_train_data=args.pre_train_data)
        tasker_shadow = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim, pre_train_data=args.pre_train_data)

    # 1. train target prompt and shadow prompt
    if args.task == 'NodeTask':
        test_acc, f1, roc, prc, prompt, answering, mia_features, mia_labels = tasker_target.run(flag='target', mia_risk=True)
        # _, test_acc_shadow, std_test_acc_shadow, f1_shadow, std_f1_shadow, roc_shadow, std_roc_shadow, _, _, prompt_shadow, answering_shadow= tasker_shadow.run(flag='shadow')
    elif args.task == 'GraphTask':
        test_acc, f1, roc, prc, prompt, answering = tasker_target.run(flag='target')
    else:
        _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _, prompt, answering= tasker.run(flag='None', mia_risk=False)
    return mia_features, mia_labels
    
    # 2. train attack model using the shadow prompt and shadow training/testing datasets, and then evaluate the attack performance
    # 2.1 train shadow prompt and the attack model
    # 2.2 train target prompt and evaluate the attack performanc

    # print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc)) 
    # print("Final F1 {:.4f}±{:.4f}(std)".format(f1,std_f1)) 
    # print("Final AUROC {:.4f}±{:.4f}(std)".format(roc, std_roc)) 

if __name__ == "__main__":
    main()



    




