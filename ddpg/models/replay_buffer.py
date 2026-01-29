import numpy as np
import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Initialize the buffer with zeros
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        # Store the transition in the buffer
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, device=device):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.state[indices]),
            torch.FloatTensor(self.action[indices]),
            torch.FloatTensor(self.next_state[indices]),
            torch.FloatTensor(self.reward[indices]),
            torch.FloatTensor(self.done[indices])
        )