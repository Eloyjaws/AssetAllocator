import torch
import numpy as np
import random 
from collections import deque, namedtuple

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    Original paper can be found at https://arxiv.org/abs/1906.04594

    This implementation was adapted from https://github.com/BY571/Normalized-Advantage-Function-NAF-
    
    """

    def __init__(self, buffer_size, batch_size, device, seed, gamma):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma

        self.n_step_buffer = deque(maxlen=1)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Params
        =====
        state (array_like): current state
        action (array_like): current action
        reward (array_like): reward for current state and action pair
        next_state (array_like): next state
        done(array_like): current end status    
        """

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == 1:
            state, action, reward, next_state, done = self.calc_multistep_return()

            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
    
    def calc_multistep_return(self):
        """
        Helper method to compute multistep returns
        """
        Return = 0
        for idx in range(1):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]
        
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
    
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)