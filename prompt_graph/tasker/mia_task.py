import torch
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva, MultiGpromptEva
from prompt_graph.pretrain import GraphPrePrompt, NodePrePrompt, prompt_pretrain_sample
from .task import BaseTask
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save,GraphDataset
from prompt_graph.evaluation import GpromptEva, AllInOneEva
import pickle
import os
from prompt_graph.utils import process
import ipdb
from .attack_model import Net
import torch.nn as nn

warnings.filterwarnings("ignore")

class MIATask(BaseTask):
      def __init__(self, data, input_dim, output_dim, prompt, graphs_list = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'MIATask'
            if self.prompt_type == 'MultiGprompt':
                  self.load_multigprompt_data()
            else:
                  self.data = data
                  if self.dataset_name == 'ogbn-arxiv':
                        self.data.y = self.data.y.squeeze()
                  self.input_dim = input_dim
                  self.output_dim = output_dim
                  self.graphs_list = graphs_list
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device) 
                  self.attack_model = Net(output_dim)
                  self.prompt = prompt
            

      def load_multigprompt_data(self):
            adj, features, labels = process.load_data(self.dataset_name)
            # adj, features, labels = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            self.output_dim = labels.shape[1]
            print('a',self.output_dim)
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)            

      
      def load_data(self):
            self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)

      def attack_train(self, trainloader, optimizer, epochs):
            criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() # cross entropy loss
            # train ntwk
            # Decay LR by a factor of 0.1 every 7 epochs
            # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            for e in range(epochs):
                  running_loss = 0
                  train_accuracy = 0
                  # This is features, labels cos we dont care about nodeID during training! only during test
                  count = 0
                  train_accuracy = 0
                  train_loss = 0
                  for features, labels in trainloader:
                        # ipdb.set_trace()
                        self.attack_model.train()
                        self.attack_model.to(self.device)
                        labels = labels.type(torch.LongTensor)
                        features, labels = features.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        # flatten features
                        features = features.view(features.shape[0], -1)

                        logps = self.attack_model(features)  # log probabilities
                        # print("labelsssss", labels.shape)
                        loss = criterion(logps, labels)

                        # Actual probabilities
                        ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss
                        top_p, top_class = ps.topk(1, dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                        # print(top_p)
                        equals = top_class == labels.view(*top_class.shape)  # making the shape of the label and top class the same
                        train_accuracy += torch.mean(equals.type(torch.FloatTensor))
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        running_loss += loss.item()
                        count += 1
                  train_accuracy = train_accuracy / count
                  train_loss = running_loss / count
                  print("Epochs: {}, train_acc: {}, train_loss: {}".format(e, train_accuracy, train_loss))
            # return train_accuracy, train_loss
      def attack_eval(self, testloader):
            criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() # cross entropy loss

            count = 0
            test_accuracy = 0
            test_loss = 0
            for features, labels in testloader:
                  # ipdb.set_trace()
                  self.attack_model.eval()
                  labels = labels.type(torch.LongTensor)
                  features, labels = features.to(self.device), labels.to(self.device)
                  # flatten features
                  features = features.view(features.shape[0], -1)
                  logps = self.attack_model(features)  # log probabilities
                  # print("labelsssss", labels.shape)
                  loss = criterion(logps, labels)
                  # Actual probabilities
                  ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss
                  top_p, top_class = ps.topk(1, dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                  # print(top_p)
                  equals = top_class == labels.view(*top_class.shape)  # making the shape of the label and top class the same
                  test_accuracy += torch.mean(equals.type(torch.FloatTensor))
                  test_loss += loss.item()
                  count += 1
            test_accuracy = test_accuracy / count
            test_loss = test_loss / count
            print("Test_acc: {}, test_loss: {}".format(test_accuracy, test_loss))
            return test_accuracy

      def run(self):
            # for all-in-one and Gprompt we use k-hop subgraph, but when wo search for best parameter, we load inducedd graph once cause it costs too much time
            # if (self.search == False) and (self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']):
            #       self.load_induced_graph()
            if self.prompt_type == 'All-in-one':
                  self.answer_epoch = 50
                  self.prompt_epoch = 50
                  self.epochs = int(self.epochs/self.answer_epoch)
            if self.use_different_dataset:
                  folder = "./Experiment_diff_dataset/sample_data/Node/{}/{}_shot".format(self.dataset_name, self.shot_num)
            else:
                  folder = "./Experiment/sample_data/Node/{}/{}_shot".format(self.dataset_name, self.shot_num)

            for i in range(1, 2): # specify the seed
                  self.initialize_gnn()
                  # self.initialize_optimizer()
                  idx_train = torch.load("{}/{}/train_idx.pt".format(folder, i)).type(torch.long).to(self.device)
                  print('idx_train',idx_train)
                  train_lbls = torch.load("{}/{}/train_labels.pt".format(folder, i)).type(torch.long).squeeze().to(self.device)
                  print("true",i,train_lbls)
                  idx_test = torch.load("{}/{}/test_idx.pt".format(folder, i)).type(torch.long).to(self.device)
                  test_lbls = torch.load("{}/{}/test_labels.pt".format(folder, i)).type(torch.long).squeeze().to(self.device)

                  # 1. split idx_train, train_lbls, idx_test, test_lbls into two equal parts as the target/shadow datasets
                  idx_train = idx_train[int(len(idx_train)/2):]
                  train_lbls = train_lbls[int(len(train_lbls)/2):]
                  idx_test = idx_test[int(len(idx_test)/2):]
                  test_lbls = test_lbls[int(len(test_lbls)/2):]

                  if len(idx_train) < len(idx_test):
                        idx_test = idx_test[:len(idx_train)]
                        test_lbls = test_lbls[:len(test_lbls)]
                  print("number of train data: {}, test data: {}".format(len(idx_train), len(idx_test)))
                  
                  if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                        train_graphs = []
                        test_graphs = []
                        # self.graphs_list.to(self.device)
                        print('distinguishing the train dataset and test dataset...')
                        for graph in self.graphs_list:                              
                              if graph.index in idx_train:
                                    train_graphs.append(graph)
                              elif graph.index in idx_test:
                                    test_graphs.append(graph)
                        print('Done!!!')

                        train_dataset = GraphDataset(train_graphs)
                        test_dataset = GraphDataset(test_graphs)

                        # 创建数据加载器
                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                        print("prepare induce graph data is finished!")

                  if self.prompt_type == 'MultiGprompt':
                        embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                        pretrain_embs = embeds[0, idx_train]
                        test_embs = embeds[0, idx_test]
                  
                  # build training dataset for attach model
                  if self.prompt_type == 'None':
                        test_acc, f1, roc, prc = GNNNodeEva(self.data, idx_test, self.gnn, self.answering,self.output_dim, self.device)                           
                  elif self.prompt_type == 'GPPT':
                        _, _, _, _, outs_train = GPPTEva(self.data, idx_train, self.gnn, self.prompt, self.output_dim, self.device)                
                        test_acc, f1, roc, prc, outs_test = GPPTEva(self.data, idx_test, self.gnn, self.prompt, self.output_dim, self.device)                
                  elif self.prompt_type == 'All-in-one':
                        _, _, _, _, outs_train = AllInOneEva(train_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)               
                        _, _, _, _, outs_test = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)                                           
                            
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        # ipdb.set_trace()
                        _, _, _, _, outs_train = GPFEva(train_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)                                                         
                        _, _, _, _, outs_test = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.output_dim, self.device)                                                         
                  elif self.prompt_type =='Gprompt':
                        _, _, _, _, outs_train = GpromptEva(train_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                        _, _, _, _, outs_test = GpromptEva(test_loader, self.gnn, self.prompt, center, self.output_dim, self.device)
                  elif self.prompt_type == 'MultiGprompt':
                        prompt_feature = self.feature_prompt(self.features)
                        test_acc, f1, roc, prc = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj, self.output_dim, self.device)  
                  
                  # train attack model
                  pos_labels = torch.ones(outs_train.shape[0])
                  neg_labels = torch.zeros(outs_test.shape[0])
                  outs_data = torch.cat((outs_train, outs_test), dim=0)
                  outs_labels = torch.cat((pos_labels, neg_labels), dim = 0)
                  # ipdb.set_trace()
                  # the number of testing dataset is significantly higher than that of the training dataset --> adjust the number of training dataset and testing dataset for the shadow prompt

                  outs_train_data = torch.cat((outs_train[:int(len(outs_train)*0.5)], outs_test[:int(len(outs_test)*0.5)]), dim=0)
                  outs_test_data = torch.cat((outs_train[int(len(outs_train)*0.5):], outs_test[int(len(outs_test)*0.5):]), dim=0)
                  outs_train_labels = torch.cat((pos_labels[:int(len(outs_train)*0.5)], neg_labels[:int(len(outs_test)*0.5)]), dim=0)
                  outs_test_labels = torch.cat((pos_labels[int(len(outs_train)*0.5):], neg_labels[int(len(outs_test)*0.5):]), dim=0)

                  attack_train_data = torch.utils.data.TensorDataset(outs_train_data, outs_train_labels) 
                  attack_test_data = torch.utils.data.TensorDataset(outs_test_data, outs_test_labels) 
                  attack_train_data_loader = torch.utils.data.DataLoader(attack_train_data, batch_size=32, shuffle=True)
                  attack_test_data_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=32, shuffle=False)

                  optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=0.01)  # 0.01 #0.00001
                  self.attack_train(attack_train_data_loader, optimizer, epochs=100)
                  test_accuracy = self.attack_eval(attack_test_data_loader)

            return  self.attack_model, test_accuracy


