import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from torch.optim import Adam
from prompt_graph.data import load4node, load4graph
import torchmetrics

class Supervised_Train(torch.nn.Module):
    def __init__(self, graph_list, input_dim, gnn_type='TransformerConv', dataset_name = 'Cora', hid_dim = 128, gln = 2, num_epoch = 1000, device : int = 5, seed: int=0):
        super().__init__()
        # self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_list = graph_list
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim =hid_dim
        self.learning_rate = 0.001
        self.weight_decay = 0.00005
        self.seed = seed

    def initialize_gnn(self):
        if self.gnn_type == 'GAT':
                self.gnn = GAT(input_dim = input_dim, hid_dim = hid_dim, out_dim = out_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GCN':
                self.gnn = GCN(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer) # output hid_dim feature
        elif self.gnn_type == 'GraphSAGE':
                self.gnn = GraphSAGE(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GIN':
                self.gnn = GIN(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GCov':
                self.gnn = GCov(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
                self.gnn = GraphTransformer(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def load_data(self):
        self.data, self.input_dim, self.out_dim = load4node(self.dataset_name)
    
    def train(self, train_idx):
        self.gnn.train()
        self.optimizer.zero_grad() 
        out = self.gnn(self.data.x, self.data.edge_index, batch=None) 
        loss = self.criterion(out[train_idx], self.data.y[train_idx])
        loss.backward()  
        self.optimizer.step()  

        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.out_dim).to(self.device)
        accuracy.reset()
        pred = out.argmax(dim=1) 
        acc = accuracy(pred[train_idx], self.data.y[train_idx])
        return loss.item(), acc

    def test(self, idx_test):
        self.gnn.eval()
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.out_dim).to(self.device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.out_dim, average="macro").to(self.device)
        auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=self.out_dim).to(self.device)
        auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=self.out_dim).to(self.device)

        accuracy.reset()
        macro_f1.reset()
        auroc.reset()
        auprc.reset()

        out = self.gnn(self.data.x, self.data.edge_index, batch=None)
        pred = out.argmax(dim=1) 
        loss = self.criterion(out[idx_test], self.data.y[idx_test])

        acc = accuracy(pred[idx_test], self.data.y[idx_test])
        f1 = macro_f1(pred[idx_test], self.data.y[idx_test])
        roc = auroc(out[idx_test], self.data.y[idx_test]) 
        prc = auprc(out[idx_test], self.data.y[idx_test]) 
        return acc.item(), f1.item(), roc.item(), prc.item(), loss.item()

    def supervised_learning(self):
        self.initialize_gnn()

        folder = "./Experiment/sample_data/Node/{}/{}_shot".format(self.dataset_name, self.shot_num)
        idx_train = torch.load("{}/{}/train_idx.pt".format(folder, i)).type(torch.long).to(self.device)
        print('idx_train',idx_train)
        train_lbls = torch.load("{}/{}/train_labels.pt".format(folder, i)).type(torch.long).squeeze().to(self.device)
        print("true",i,train_lbls)
        idx_test = torch.load("{}/{}/test_idx.pt".format(folder, i)).type(torch.long).to(self.device)
        test_lbls = torch.load("{}/{}/test_labels.pt".format(folder, i)).type(torch.long).squeeze().to(self.device)

        train_loss_min = 1000000
        patience = 20
        cnt_wait = 0

        file_path = f"./Experiment/supervised_results/{self.dataset_name}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        for epoch in range(1, self.epochs + 1):
            time0 = time.time()
            train_loss, train_acc = self.train(idx_train)
            test_acc, f1, roc, prc, test_loss = self.test(idx_test)
            print("***epoch: {}/{} | train_loss: {:.4} | train_acc: {:.4} | test_loss: {:.4} | test_acc: {:.4}".format(epoch, self.epochs , train_loss, train_acc, test_loss, test_acc))

            filename = "{}.{}hidden_dim.seed{}.txt".format(self.gnn_type, str(self.hid_dim), self.seed)
            save_path = os.path.join(file_path, filename)
            # if save_path already exist, clear all existing contents
            if (epoch == 1) and (os.path.exists(save_path)): 
                os.remove(save_path) 
            with open(save_path, 'a') as f:
                f.write('%d %.8f %.8f %.8f %.8f'%(epoch, train_loss, test_loss, train_acc, test_acc))
                f.write("\n")

            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break



        folder_path = f"./Experiment/supervised_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.gnn.state_dict(),
                    "{}/{}.{}.pth".format(folder_path, self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
        print("+++model saved ! {}/{}.{}.pth".format(self.dataset_name, self.gnn_type, str(self.hid_dim) + 'hidden_dim'))


        
#     def load_node_data(self):
#         self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
#         self.data.to(self.device)
#         self.input_dim = self.dataset.num_features
#         self.output_dim = self.dataset.num_classes

