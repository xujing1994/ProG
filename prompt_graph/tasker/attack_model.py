import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    # define nn
    def __init__(self, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(output_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # print("attack X",X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)

        return X
    
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)
