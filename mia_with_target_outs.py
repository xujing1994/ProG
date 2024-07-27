
from prompt_graph.tasker.attack_model import Net
import torch.nn as nn
import torch
from prompt_graph.utils import get_args
import os
import numpy as np
import ipdb
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from prompt_graph.utils import seed_everything

args = get_args()
Seeds = range(100)
print(os.getcwd())
seed_everything(args.seed)
pre_train_type = ['DGI', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphMAE']
pre_train_data = ['CiteSeer', 'ogbn-arxiv', 'Photo', 'PubMed']
prompt_type = ['GPF-plus']

def read_data(path, seeds):
    print(path)
    d = np.zeros(((len(seeds), int(args.shot_num*7), 8))) # [len(Seeds), shot_num*num_classes, 8]
    count = np.zeros(len(seeds))
    with open(path, 'r') as f:
        for line in f.readlines()[:]:
            line = line.replace('\n', '')
            data = line.split(' ')
            seed = int(data[0])
            if len(seeds) == 1:
                seed = int(seed-42)
            d[seed, int(count[seed])] = data[1:]
            count[seed] += 1
    return d

def train(attack_model, trainloader, optimizer, epoch, device):
    criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() # cross entropy loss
    # train ntwk
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    running_loss = 0
    train_accuracy = 0
    # This is features, labels cos we dont care about nodeID during training! only during test
    count = 0
    train_accuracy = 0
    train_loss = 0
    for features, labels in trainloader:
        attack_model.train()
        attack_model.to(device)
        labels = labels.type(torch.LongTensor)
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        # flatten features
        features = features.view(features.shape[0], -1)

        logps = attack_model(features)  # log probabilities
        # print("labelsssss", labels.shape)
        loss = criterion(logps, labels)

        # Actual probabilities
        ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss
        top_p, top_class = ps.topk(1, dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
        # print(top_p)
        equals = top_class == labels.view(*top_class.shape)  # making the shape of the label and top class the same
        train_accuracy += torch.mean(equals.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count += 1
    train_accuracy = train_accuracy / count
    train_loss = running_loss / count
    print("Epochs: {}, train_acc: {}, train_loss: {}".format(epoch, train_accuracy, train_loss))
    return train_accuracy, train_loss

def eval(attack_model, testloader, device):
    criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() # cross entropy loss

    count = 0
    test_accuracy = 0
    test_loss = 0
    labels_ = []
    prob_class1_ = []
    for features, labels in testloader:
            # ipdb.set_trace()
            attack_model.eval()
            labels = labels.type(torch.LongTensor)
            features, labels = features.to(device), labels.to(device)
            labels_.append(labels)
            # flatten features
            features = features.view(features.shape[0], -1)
            logps = attack_model(features)  # log probabilities
            # print("labelsssss", labels.shape)
            loss = criterion(logps, labels)
            # Actual probabilities
            ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss
            top_p, top_class = ps.topk(1, dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
            prob_class1_.append(logps[:, 1])
            # print(top_p)
            equals = top_class == labels.view(*top_class.shape)  # making the shape of the label and top class the same
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))
            test_loss += loss.item()
            count += 1

    test_accuracy = test_accuracy / count
    # calculate fpr, tpr
    labels = torch.cat(labels_, dim=0)
    prob_class1 = torch.cat(prob_class1_, dim=0)
    # ipdb.set_trace()
    fpr, tpr, _ = metrics.roc_curve(labels.detach().numpy(), prob_class1.detach().numpy())
    test_loss = test_loss / count
    print("Test_acc: {}, test_loss: {}".format(test_accuracy, test_loss))
    return test_accuracy, fpr, tpr

def membership_inference_attack(train_data, test_data, target_train_data, target_test_data, num_classes, device):
    attack_model = Net(output_dim=7)
    feature_train = torch.from_numpy(train_data[:, :, :-1].reshape((int(len(Seeds)*num_classes*args.shot_num), 7))).float()
    feature_test = torch.from_numpy(test_data[:, :, :-1].reshape((int(len(Seeds)*num_classes*args.shot_num), 7))).float() # 7000*7

    target_feature_train = torch.from_numpy(target_train_data[:, :, :-1].reshape((int(1*num_classes*args.shot_num), 7))).float()
    target_feature_test = torch.from_numpy(target_test_data[:, :, :-1].reshape((int(1*num_classes*args.shot_num), 7))).float()

    # select top-5 features as the attack feature
    # ipdb.set_trace()
    # feature_train = torch.from_numpy(np.sort(train_data[:, :, :-1].reshape(int(len(Seeds)*70), 7), axis=1)[:, ::-1][:, :5].copy()).float()
    # feature_test = torch.from_numpy(np.sort(test_data[:, :, :-1].reshape(int(len(Seeds)*70), 7), axis=1)[:, ::-1][:, :5].copy()).float()
    # labels_train = torch.from_numpy(train_data[:, :, -1].astype(np.int64).flatten()) # 7000
    # labels_test = torch.from_numpy(test_data[:, :, -1].astype(np.int64).flatten())
    pos_labels = torch.ones(feature_train.shape[0])
    neg_labels = torch.zeros(feature_test.shape[0])

    target_pos_labels = torch.ones(target_feature_train.shape[0])
    target_neg_labels = torch.ones(target_feature_test.shape[0])

    attack_train = torch.cat((feature_train[:int(len(feature_train)*0.8)], feature_test[:int(len(feature_test)*0.8)]), dim=0)
    # attack_test = torch.cat((target_feature_train, target_feature_test), dim=0)
    attack_test = torch.cat((feature_train[int(len(feature_train)*0.8):], feature_test[int(len(feature_test)*0.8):]), dim=0)
    labels_attack_train = torch.cat((pos_labels[:int(len(pos_labels)*0.8)], neg_labels[:int(len(neg_labels)*0.8)]), dim=0)
    # labels_attack_test = torch.cat((target_pos_labels, target_neg_labels), dim=0)
    labels_attack_test = torch.cat((pos_labels[int(len(pos_labels)*0.8):], neg_labels[int(len(neg_labels)*0.8):]), dim=0)

    attack_train_data = torch.utils.data.TensorDataset(attack_train, labels_attack_train) 
    attack_test_data = torch.utils.data.TensorDataset(attack_test, labels_attack_test) 
    attack_train_dataloader = torch.utils.data.DataLoader(attack_train_data, batch_size=32, shuffle=True)
    attack_test_dataloader = torch.utils.data.DataLoader(attack_test_data, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=7e-3)  # 0.01 #0.00001
    epochs = 600
    for e in range(epochs):
        train(attack_model=attack_model, trainloader=attack_train_dataloader, optimizer=optimizer, epoch=e, device=device)
    # calculate the false positive rate and true positive rate
    # read target_outs as the attack testing data
    
    test_accuracy, fpr, tpr = eval(attack_model, attack_test_dataloader, device=device)
    # roc_auc = metrics.auc(fpr, tpr)
    # # draw auc-roc curve
    # fig, ax = plt.subplots()
    # ax.set_title('Receiver Operating Characteristic')
    # ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # ax.legend(loc = 'lower right')
    # ax.plot([0, 1], [0, 1],'r--')
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.set_ylabel('True Positive Rate')
    # ax.set_xlabel('False Positive Rate')
    # save_path = "./figs/AUC_ROC/{}shot/".format(args.shot_num) + '{}_{}_{}_{}_{}.png'.format(args.pre_train_type, args.prompt_type, args.pre_train_data, args.dataset_name, args.gnn_type)
    # print(save_path)
    # isExist = os.path.exists(os.path.split(save_path)[0])
    # if not isExist:
    #     os.makedirs(os.path.split(save_path)[0])
    # plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1200)

def main():
    num_classes = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # for all-in-one and Gprompt we use k-hop subgraph, but when wo search for best parameter, we load inducedd graph once cause it costs too much time
    # if (self.search == False) and (self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']):
    #       self.load_induced_graph()

    # load training and testing features
    for ptt in pre_train_type[-1:]:
        for ptd in pre_train_data[:1]:
            for prompt in prompt_type:
                for shot_num in range(1, 2):
                    args.shot_num = shot_num
                    args.pre_train_type = ptt
                    args.pre_train_data = ptd
                    args.prompt_type = prompt
                    if args.use_different_dataset:
                        folder = "./Experiment_diff_dataset/outs/Node/{}shot/{}_{}".format(shot_num, args.dataset_name, ptd)
                        target_folder = "./Experiment_diff_dataset/target_outs/Node/{}shot/{}_{}".format(shot_num, args.dataset_name, ptd)
                    else:
                        folder = "./Experiment/outs/Node/{}shot/{}_{}".format(shot_num, args.dataset_name, ptd)
                        target_folder = "./Experiment/target_outs/Node/{}shot/{}_{}".format(shot_num, args.dataset_name, ptd)
                    train_data = read_data(os.path.join(folder, 'train/{}_{}_{}.txt'.format(ptt, prompt, args.gnn_type)), seeds=Seeds)
                    test_data = read_data(os.path.join(folder, 'test/{}_{}_{}.txt'.format(ptt, prompt, args.gnn_type)), seeds=Seeds)
                    target_train_data = read_data(os.path.join(target_folder, 'train/{}_{}_{}.txt'.format(ptt, prompt, args.gnn_type)), seeds=range(1))
                    target_test_data = read_data(os.path.join(target_folder, 'test/{}_{}_{}.txt'.format(ptt, prompt, args.gnn_type)), seeds=range(1))

                    membership_inference_attack(train_data, test_data, target_train_data, target_test_data, num_classes, device)

    # return  attack_model, test_accuracy

if __name__ == '__main__':
      main()
