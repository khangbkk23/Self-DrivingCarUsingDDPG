import torch
import torch.nn as nn
import torch.nn.functional as F
from perception.models import CNNEncoder

class Actor(nn.Module):
    def __init__(self, action_dim, max_action, config):
        super(Actor, self).__init__()
        self.encoder = CNNEncoder(config)
        self.fc1 = nn.Linear(self.encoder.flatten_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        features = self.encoder(state)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.out(x)) * self.max_action
        return action