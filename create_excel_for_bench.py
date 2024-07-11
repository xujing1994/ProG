import pandas as pd
import os

graph_dataset_name = ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'ogbg-ppa','DD']
node_dataset_name = ['PubMed', 'CiteSeer', 'Cora', 'Wisconsin', 'Texas', 'ogbn-arxiv', 'Actor', 'Flickr']
shot_nums = [1,2,3,4,5,6,7,8,9,10]
for dataset_name in node_dataset_name:
    for shot_num in shot_nums:
        pre_train_types = ['None', 'DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
        prompt_types = ['None', 'GPPT', 'All-in-one', 'Gprompt', 'GPF', 'GPF-plus']


        column_names = [f"{pre_train}+{prompt}"  for prompt in prompt_types for pre_train in pre_train_types if pre_train != 'None' or prompt == 'None']

        # 创建DataFrame
        data = pd.DataFrame(columns=column_names, index=['Final Accuracy', 'Final F1', 'Final AUROC'])

        gnn_type = ['GCN', 'GAT', 'GCov', 'GraphSAGE', 'GIN', 'GraphTransformer']
        for i, gt in enumerate(gnn_type):
            file_name = gnn_type[i] +"_total_results.xlsx"
            file_path = os.path.join('./Experiment/ExcelResults/Node/'+str(shot_num)+'shot/'+ dataset_name +'/', file_name)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)        
            data.to_excel(file_path)
            text_filepath = os.path.join('./Experiment/ExcelResults/Node/'+str(shot_num)+'shot/'+ dataset_name +'/', gnn_type[i] +"_total_results.txt")
            with open(text_filepath, 'w') as f:
                f.write("pre_train+prompt Learning_rate Weight_decay Batch_size Epochs shot_num hid_dim seed target_Final_Accuracy target_Final_F1 target_Final_AUROC shadow_Final_Accuracy shadow_Final_F1 shadow_Final_AUROC MIA_ASR")
                f.write("\n")

            # 打印信息确认文件已保存
            print(f"Data saved to {file_path} successfully.")

        for i, gt in enumerate(gnn_type):
            file_name = gnn_type[i] +"_total_results.xlsx"
            file_path = os.path.join('./Experiment_diff_dataset/ExcelResults/Node/'+str(shot_num)+'shot/'+ dataset_name +'/', file_name)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)        
            data.to_excel(file_path)
            text_filepath = os.path.join('./Experiment_diff_dataset/ExcelResults/Node/'+str(shot_num)+'shot/'+ dataset_name +'/', gnn_type[i] +"_total_results.txt")
            with open(text_filepath, 'w') as f:
                f.write("pre_train+prompt Learning_rate Weight_decay Batch_size Epochs shot_num hid_dim seed target_Final_Accuracy target_Final_F1 target_Final_AUROC shadow_Final_Accuracy shadow_Final_F1 shadow_Final_AUROC MIA_ASR")
                f.write("\n")

            # 打印信息确认文件已保存
            print(f"Data saved to {file_path} successfully.")

for dataset_name in graph_dataset_name:
    for shot_num in shot_nums:
        pre_train_types = ['None', 'DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
        prompt_types = ['None', 'GPPT', 'All-in-one', 'Gprompt', 'GPF', 'GPF-plus']


        column_names = [f"{pre_train}+{prompt}"  for prompt in prompt_types for pre_train in pre_train_types if pre_train != 'None' or prompt == 'None']

        # 创建DataFrame
        data = pd.DataFrame(columns=column_names, index=['Final Accuracy', 'Final F1', 'Final AUROC'])

        gnn_type = ['GCN', 'GAT', 'GCov', 'GraphSAGE', 'GIN', 'GraphTransformer']
        for i, gt in enumerate(gnn_type):
            file_name = gnn_type[i] +"_total_results.xlsx"
            file_path = os.path.join('./Experiment/ExcelResults/Graph/'+str(shot_num)+'shot/'+ dataset_name +'/', file_name)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)        
            data.to_excel(file_path)

            text_filepath = os.path.join('./Experiment/ExcelResults/Graph/'+str(shot_num)+'shot/'+ dataset_name +'/', gnn_type[i] +"_total_results.txt")
            with open(text_filepath, 'w') as f:
                f.write("pre_train+prompt Learning_rate Weight_decay Batch_size Epochs shot_num hid_dim seed target_Final_Accuracy target_Final_F1 target_Final_AUROC shadow_Final_Accuracy shadow_Final_F1 shadow_Final_AUROC MIA_ASR")
                f.write("\n")        
            # 打印信息确认文件已保存
            print(f"Data saved to {file_path} successfully.")

        for i, gt in enumerate(gnn_type):
            file_name = gnn_type[i] +"_total_results.xlsx"
            file_path = os.path.join('./Experiment_diff_dataset/ExcelResults/Graph/'+str(shot_num)+'shot/'+ dataset_name +'/', file_name)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)        
            data.to_excel(file_path)

            text_filepath = os.path.join('./Experiment_diff_dataset/ExcelResults/Graph/'+str(shot_num)+'shot/'+ dataset_name +'/', gnn_type[i] +"_total_results.txt")
            with open(text_filepath, 'w') as f:
                f.write("pre_train+prompt Learning_rate Weight_decay Batch_size Epochs shot_num hid_dim seed target_Final_Accuracy target_Final_F1 target_Final_AUROC shadow_Final_Accuracy shadow_Final_F1 shadow_Final_AUROC MIA_ASR")
                f.write("\n")        
            # 打印信息确认文件已保存
            print(f"Data saved to {file_path} successfully.")