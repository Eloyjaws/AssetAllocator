import torch
import numpy as np
import random

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.idx = 0

    def store(self, experience):
        index = self.idx % self.capacity
        self.buffer[index] = experience
        self.idx += 1

    def sample(self, batch_size, device):
        experience = np.array(random.sample(self.buffer[:self.idx], batch_size), dtype=object)

        states = torch.tensor(np.array(experience[:, 0].tolist()), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(experience[:, 1].tolist()), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(experience[:, 2].tolist()), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(experience[:, 3].tolist()), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(experience[:, 4].tolist())).to(device)

        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        if self.idx <= self.capacity:
            return self.idx
        return self.capacity