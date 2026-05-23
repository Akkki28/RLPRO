import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=input_len,out_features=64)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(in_features=64,out_features=128)
        self.output = nn.Linear(in_features=128,out_features=output_len)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden(x))
        x = self.output(x)
        x = self.softmax(x)
        return x