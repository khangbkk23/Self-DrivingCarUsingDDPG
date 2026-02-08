import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, device):
        indices = np.random.choice(self.size, batch_size, replace=False)

        states = torch.FloatTensor(self.state[indices]).to(device) / 255.0
        actions = torch.FloatTensor(self.action[indices]).to(device)
        next_states = torch.FloatTensor(self.next_state[indices]).to(device) / 255.0
        rewards = torch.FloatTensor(self.reward[indices]).to(device)
        dones = torch.FloatTensor(self.done[indices]).to(device)

        return states, actions, next_states, rewards, dones