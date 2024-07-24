
from prompt_graph.tasker.attack_model import Net
import torch.nn as nn
import torch
from prompt_graph.utils import get_args
import os
import numpy as np
import ipdb

args = get_args()
Seeds = range(100)
print(os.getcwd())

def read_data(path):
    d = np.zeros(((len(Seeds), 70, 8))) # [len(Seeds), shot_num*num_classes, 8]
    count = np.zeros(len(Seeds))
    with open(path, 'r') as f:
        for line in f.readlines()[:]:
            line = line.replace('\n', '')
            data = line.split(' ')
            seed = int(data[0])
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
    for features, labels in testloader:
            # ipdb.set_trace()
            attack_model.eval()
            labels = labels.type(torch.LongTensor)
            features, labels = features.to(device), labels.to(device)
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
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))
            test_loss += loss.item()
            count += 1
    test_accuracy = test_accuracy / count
    test_loss = test_loss / count
    print("Test_acc: {}, test_loss: {}".format(test_accuracy, test_loss))
    return test_accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # for all-in-one and Gprompt we use k-hop subgraph, but when wo search for best parameter, we load inducedd graph once cause it costs too much time
    # if (self.search == False) and (self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']):
    #       self.load_induced_graph()
    attack_model = Net(output_dim=7)
    if args.use_different_dataset:
            folder = "./Experiment_diff_dataset/outs/Node/{}shot/{}_{}".format(args.shot_num, args.dataset_name, args.pre_train_data)
    else:
            folder = "./Experiment/outs/Node/{}shot/{}_{}".format(args.shot_num, args.dataset_name, args.pre_train_data)
    train_data = read_data(os.path.join(folder, 'train/{}_{}_{}.txt'.format(args.pre_train_type, args.prompt_type, args.gnn_type)))
    test_data = read_data(os.path.join(folder, 'test/{}_{}_{}.txt'.format(args.pre_train_type, args.prompt_type, args.gnn_type)))
    feature_train = torch.from_numpy(train_data[:, :, :-1].reshape((int(len(Seeds)*70), 7))).float()
    feature_test = torch.from_numpy(test_data[:, :, :-1].reshape((int(len(Seeds)*70), 7))).float() # 7000*7

    # select top-5 features as the attack feature
    # ipdb.set_trace()
    # feature_train = torch.from_numpy(np.sort(train_data[:, :, :-1].reshape(int(len(Seeds)*70), 7), axis=1)[:, ::-1][:, :5].copy()).float()
    # feature_test = torch.from_numpy(np.sort(test_data[:, :, :-1].reshape(int(len(Seeds)*70), 7), axis=1)[:, ::-1][:, :5].copy()).float()
    # labels_train = torch.from_numpy(train_data[:, :, -1].astype(np.int64).flatten()) # 7000
    # labels_test = torch.from_numpy(test_data[:, :, -1].astype(np.int64).flatten())
    pos_labels = torch.ones(feature_train.shape[0])
    neg_labels = torch.zeros(feature_test.shape[0])

    attack_train = torch.cat((feature_train[:int(len(feature_train)*0.5)], feature_test[:int(len(feature_test)*0.5)]), dim=0)
    attack_test = torch.cat((feature_train[int(len(feature_train)*0.5):], feature_test[int(len(feature_test)*0.5):]), dim=0)
    labels_attack_train = torch.cat((pos_labels[:int(len(pos_labels)*0.5)], neg_labels[:int(len(neg_labels)*0.5)]), dim=0)
    labels_attack_test = torch.cat((pos_labels[int(len(pos_labels)*0.5):], neg_labels[int(len(neg_labels)*0.5):]), dim=0)

    attack_train_data = torch.utils.data.TensorDataset(attack_train, labels_attack_train) 
    attack_test_data = torch.utils.data.TensorDataset(attack_test, labels_attack_test) 
    attack_train_dataloader = torch.utils.data.DataLoader(attack_train_data, batch_size=32, shuffle=True)
    attack_test_dataloader = torch.utils.data.DataLoader(attack_test_data, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=7e-3)  # 0.01 #0.00001
    epochs = 300
    for e in range(epochs):
        train(attack_model=attack_model, trainloader=attack_train_dataloader, optimizer=optimizer, epoch=e, device=device)

    test_accuracy = eval(attack_model, attack_test_dataloader, device=device)

    return  attack_model, test_accuracy

if __name__ == '__main__':
      main()
