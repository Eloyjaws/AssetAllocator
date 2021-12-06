import torch
import numpy as np
import random

class Memory:
    """This is the replay buffer class for the TD3 Agent.

    Original paper can be found at https://arxiv.org/abs/1802.09477

    This implementation was adapted from https://github.com/saashanair/rl-series/tree/master/td3
    
    """
    def __init__(self, capacity):
        """Initialize a ReplayBuffer object.

        Args:
            capacity (int): maximum size of buffer
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.idx = 0

    def store(self, experience):
        """Add a new experience to memory.

        Args:
            experience (array_like): current state, current action, reward, next state, and current end status tuple  
        """
        index = self.idx % self.capacity
        self.buffer[index] = experience
        self.idx += 1

    def sample(self, batch_size, device):
        """
        Randomly sample a batch of experiences from memory.

        Args:
            batch_size (int): Batch size to sample
            device: One of cuda or cpus
        """
        experience = np.array(random.sample(self.buffer[:self.idx], batch_size), dtype=object)

        states = torch.tensor(np.array(experience[:, 0].tolist()), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(experience[:, 1].tolist()), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(experience[:, 2].tolist()), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(experience[:, 3].tolist()), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(experience[:, 4].tolist())).to(device)

        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        """
        Return the current size of internal memory.
        """
        if self.idx <= self.capacity:
            return self.idx
        return self.capacity