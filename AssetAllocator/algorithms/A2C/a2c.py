import math
import random
import sys
sys.path.append('..')

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# helper function to convert numpy arrays to tensors
def tensor(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
    
    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        res =  torch.distributions.Normal(means, stds)
        return res
    
## Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, X):
        return self.model(X)
    
def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done)
        discounted.append(ret)
    
    return discounted[::-1]

def process_memory(memory, gamma=0.99, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []

    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)
    
    if discount_rewards:
        if False and dones[-1] == 0:
            rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        else:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions = tensor(actions)
    states = tensor(states)
    next_states = tensor(next_states)
    rewards = tensor(rewards).view(-1, 1)
    dones = tensor(dones).view(-1, 1)
    return actions, rewards, states, next_states, dones

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)
    
    
class A2CLearner():
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(memory, self.gamma, discount_rewards)

        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        value = self.critic(states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()
        
        actor_loss = (-logs_probs*advantage.detach()).mean() - entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()
        
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.critic_optim.step()
        
    def predict(self, state):
        dists = self.actor(tensor(state))
        actions = dists.sample()
        actions_clipped = torch.nn.Softmax(dim = 0)(actions).detach().data.numpy()
        return actions_clipped
    
class Runner():
    def __init__(self, env, actor):
        self.env = env
        self.actor = actor
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
    
    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state = self.env.reset()
    
    def run(self, max_steps, print_every, memory=None):
        if not memory: memory = []
        
        count_of_dones = 0
        flag = False
        for i in range(max_steps):
            if self.done: 
                self.reset()
            
            dists = self.actor(tensor(self.state))
            actions = dists.sample()
            actions_clipped = torch.nn.Softmax(dim = 0)(actions).detach().data.numpy()

            next_state, reward, self.done, info = self.env.step(actions_clipped)
            memory.append((actions_clipped, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                count_of_dones += 1
                self.episode_rewards.append(self.episode_reward)
                flag = True
                
            if flag and count_of_dones % print_every == 0:
                print(f'Score at timestep {self.steps}: {self.episode_reward}.')
                flag = False
        
        return memory