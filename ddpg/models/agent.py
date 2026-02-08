import torch
import torch.optim as optim
import os

from .actor import Actor
from .critic import Critic

class DDPGAgent:
    """
    DDPGAgent is the container for all DDPG networks and optimizers.
    It manages the interaction between the policy and the environment.
    """
    def __init__(self, action_dim, max_action, config, device):
        self.cfg = config
        self.device = device
        self.max_action = max_action
        
        # 1. Initialize networks
        # Actor networks
        self.actor = Actor(action_dim, max_action, config).to(device)
        self.actor_target = Actor(action_dim, max_action, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = Critic(action_dim, config).to(device)
        self.critic_target = Critic(action_dim, config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 2. Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=float(config['actor_lr'])
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=float(config['critic_lr'])
        )

    def select_action(self, state):

        self.actor.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # [0, 255] -> [0, 1]
        state_tensor = state_tensor / 255.0
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
            
        self.actor.train()
        return action
    
    def save(self, folder_path, filename="ddpg_model.pth"):
        os.makedirs(folder_path, exist_ok=True)
        
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        save_path = os.path.join(folder_path, filename)
        torch.save(save_dict, save_path)
        print(f"--> Model saved to {save_path}")

    def load(self, file_path):
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"--> Successfully loaded model from {file_path}")