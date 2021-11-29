"""
Script that contains the details about the experience replay buffer used to ensure training stability
"""

## initial thought was to use deque, but with a large replay memory it turns out to be very inefficient -- see https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3

import random
import numpy as np

import torch

class ReplayMemory:
    """
    Class representing the replay buffer used for storing experiences for off-policy learning
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] # create a list of lists, such that each experience added to memory is a list of 5-items of the form (state, action, next_state, reward, done)
        self.idx = 0

    def store(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity # for circular memory


    def sample(self, batchsize, device):

        transitions = np.array(random.sample(self.buffer, batchsize), dtype=object)
        states = torch.tensor(np.array(transitions[:, 0].tolist()), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(transitions[:, 1].tolist()), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(transitions[:, 2].tolist()), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(transitions[:, 3].tolist()), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(transitions[:, 4].tolist())).to(device)

        return states, actions, next_states, rewards, dones


    def __len__(self):
        return len(self.buffer)