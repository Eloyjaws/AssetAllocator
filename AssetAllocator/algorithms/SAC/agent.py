import math
import random
import sys
sys.path.append('../')

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from .SAC import *

class SACAgent:
    def __init__(self, env, hidden_dim = 256, value_lr = 3e-4, soft_q_lr = 3e-4, policy_lr = 3e-4,
                 gamma=0.99, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0.0, soft_tau=1e-2,
                 replay_buffer_size = 1_000_000, batch_size = 128, device = 'cpu'):
        
        self.env = env
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim  = self.env.observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self.device = device

        self.value_lr  = value_lr
        self.soft_q_lr = soft_q_lr
        self.policy_lr = policy_lr
        self.replay_buffer_size = replay_buffer_size
        
        self.gamma = gamma
        self.mean_lambda = mean_lambda
        self.std_lambda = std_lambda
        self.z_lambda = z_lambda
        self.soft_tau = soft_tau

        self.value_net        = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, hidden_dim).to(self.device)

        self.soft_q_net = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim,
                                        self.hidden_dim, device = self.device).to(self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)


        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.batch_size = batch_size
        
    def soft_q_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)


        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()


        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss  = self.std_lambda  * log_std.pow(2).mean()
        z_loss    = self.z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
    
    def learn(self, timesteps, print_every = 100):
        idx = 0
        flag = False
        count_of_dones = 0
        
        while idx < timesteps:
            state = self.env.reset()
            ep_reward = 0
            done = False
                
            while not done:
                action = self.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.push(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) > self.batch_size:
                    self.soft_q_update()

                state = next_state
                ep_reward += reward
                idx += 1

                if done:
                    count_of_dones += 1
                    flag = True

                if flag and count_of_dones % print_every == 0:
                        print(f'Score at timestep {idx}: {ep_reward}.')
                        flag = False
                
                if idx > timesteps:
                    break
                        
    def predict(self, state):
        action = self.policy_net.get_action(state)
        return action
    
    def save(self, filename):
        torch.save(self.value_net.state_dict(), filename + '_value_net')
        torch.save(self.value_optimizer.state_dict(), filename + '_value_optimizer')

        torch.save(self.soft_q_net.state_dict(), filename + '_soft_q_net')
        torch.save(self.soft_q_optimizer.state_dict(), filename + '_soft_q_optimizer')

        torch.save(self.policy_net.state_dict(), filename + '_policy_net')
        torch.save(self.policy_optimizer.state_dict(), filename + '_policy_optimizer')

    def load(self, filename):
        self.value_net.load_state_dict(torch.load(filename + '_value_net'))
        self.value_optimizer.load_state_dict(torch.load(filename + '_value_optimizer'))

        self.soft_q_net.load_state_dict(torch.load(filename + '_soft_q_net'))
        self.soft_q_optimizer.load_state_dict(torch.load(filename + '_soft_q_optimizer'))

        self.policy_net.load_state_dict(torch.load(filename + '_policy_net'))
        self.policy_optimizer.load_state_dict(torch.load(filename + '_policy_optimizer'))
        
        self.target_value_net.load_state_dict(self.value_net.state_dict())
