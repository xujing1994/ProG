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
seed_everything(args.seed)
print('dataset_name', args.dataset_name)

if __name__ == "__main__":
    if args.use_different_dataset:
        args.pre_train_model_path = './Experiment_diff_dataset/pre_trained_model/{}/{}.{}.128hidden_dim.pth'.format(args.pre_train_data, args.pre_train_type, args.gnn_type)
    else:
        args.pre_train_model_path = './Experiment/pre_trained_model/{}/{}.{}.128hidden_dim.pth'.format(args.pre_train_data, args.pre_train_type, args.gnn_type)
    # pretrain_path = args.pre_train_model_path
    # pretrain_type = os.path.split(pretrain_path)[1].split('.')[0]
    print("Dataset: {}, Pre-train Data: {}, GNN: {}, Pretrain: {}, Prompt: {}, ShotNum: {}, Seed: {}".format(args.dataset_name, args.pre_train_data, args.gnn_type, args.pre_train_type, args.prompt_type, args.shot_num, args.seed))
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
                        batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim)
        tasker_shadow = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                        batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim)

    # 1. train target prompt and shadow prompt
    if args.task == 'NodeTask':
        _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _, prompt, answering= tasker_target.run(flag='target')
        _, test_acc_shadow, std_test_acc_shadow, f1_shadow, std_f1_shadow, roc_shadow, std_roc_shadow, _, _, prompt_shadow, answering_shadow= tasker_shadow.run(flag='shadow')
    elif args.task == 'GraphTask':
        test_acc, f1, roc, prc, prompt, answering = tasker_target.run(flag='target')
        test_acc_shadow, f1_shadow, roc_shadow, prc_shadow, prompt_shadow, answering_shadow = tasker_shadow.run(flag='shadow')
    else:
        _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _, prompt, answering= tasker.run(flag='None')
    # 2. train attack model using the shadow prompt and shadow training/testing datasets, and then evaluate the attack performance
    # 2.1 train shadow prompt and the attack model
    # 2.2 train target prompt and evaluate the attack performance
    if args.task != 'FineTuneNodeTask':
        attack_tasker = MIATask(pre_train_model_path = args.pre_train_model_path, 
                            dataset_name = args.dataset_name, num_layer = args.num_layer, prompt=prompt_shadow,
                            gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                            epochs = 100, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                            batch_size = args.batch_size, seed=args.seed, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list, use_different_dataset=args.use_different_dataset)
        attack_model, asr = attack_tasker.run()

    # ipdb.set_trace()

    # print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc)) 
    # print("Final F1 {:.4f}±{:.4f}(std)".format(f1,std_f1)) 
    # print("Final AUROC {:.4f}±{:.4f}(std)".format(roc, std_roc)) 
    # save results to txt file
    file_name = args.gnn_type +"_total_results.txt"
    if args.use_different_dataset:
        folder = "./Experiment_diff_dataset/ExcelResults"
    else:
        folder = "./Experiment/ExcelResults"
    
    if args.task == 'NodeTask':
        file_path = os.path.join('{}/Node/'.format(folder)+str(args.shot_num)+'shot/'+ args.dataset_name +'_'+args.pre_train_data + '/', file_name)
    elif args.task == 'FineTuneNodeTask':
        file_path = os.path.join('{}/FineTuneNode/'.format(folder)+str(args.shot_num)+'shot/'+ args.dataset_name +'_'+args.pre_train_data + '/', file_name)
    else:
        file_path = os.path.join('{}/Graph/'.format(folder)+str(args.shot_num)+'shot/'+ args.dataset_name +'_'+args.pre_train_data + '/', file_name)
    folder_path = os.path.split(file_path)[0]
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    if args.task == 'FineTuneNodeTask':
        test_acc_shadow, std_test_acc_shadow, f1_shadow, std_f1_shadow, roc_shadow, std_roc_shadow, asr = None, None, None, None, None, None, None

    with open(file_path, 'a') as f:
        print(file_path)
        #f.write("pre_train+prompt learning_rate weight_decay batch_size Final_Accuracy Final_F1 Final_AUROC")
        f.write("{}+{} {} {} {} {} {} {} {} {}+{} {}+{} {}+{} {}+{} {}+{} {}+{} {}".format(pre_train_type, args.prompt_type, args.lr, args.decay, args.batch_size, args.epochs, args.shot_num, args.hid_dim, args.seed, test_acc, std_test_acc, f1, std_f1, roc, std_roc, test_acc_shadow, std_test_acc_shadow, f1_shadow, std_f1_shadow, roc_shadow, std_roc_shadow, asr))
        f.write("\n")


    




