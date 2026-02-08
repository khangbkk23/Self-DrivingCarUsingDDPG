import torch
import torch.nn as nn
import torch.nn.functional as F
from perception.models import CNNEncoder

class Critic(nn.Module):
    def __init__(self, action_dim, config):
        super(Critic, self).__init__()
        self.encoder = CNNEncoder(config)
        
        self.fc1 = nn.Linear(self.encoder.flatten_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, state, action):
        features = self.encoder(state)
        
        x = torch.cat([features, action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value