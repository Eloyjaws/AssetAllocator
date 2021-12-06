import argparse
import math
import os
import numpy as np
import gym
from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils

from .reinforce_continuous import REINFORCE
from .normalized_actions import check_and_normalize_box_actions

default_device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def softmax(x, axis=0):
    # Use the LogSumExp Trick
    max_val = np.amax(x, axis=axis, keepdims=True)
    x = x - max_val

    # Softmax
    num = np.exp(x)
    denum = num.sum(axis=axis, keepdims=True)
    softmax = num/denum
    return softmax


class REINFORCEAgent:
    """
        Helper class to manage and train REINFORCE Agent
    """
    def __init__(self, env, device=default_device):

        """Initializes the REINFORCE Agent

        Args:
            env (gym object): Gym environment for the agent to interact with
            device (str, optional): One of cuda or cpu. Defaults to 'cuda'.
        """
        torch.manual_seed(env.seed)
        np.random.seed(env.seed)

        hidden_size = 128

        self.env = env
        self.gamma = 0.99
        self.agent = REINFORCE(
            hidden_size, env.observation_space.shape[0], env.action_space)

    def train(self, timesteps, print_every):
        """Helper method to train the agent

        Args:
            timesteps (int): Total timesteps the agent has interacted for
            print_every (int): Verbosity control
        """       
        reward_history = []  # tracks the reward per episode
        best_score = -np.inf

        epochs = timesteps//self.env.episode_length + 1
        total_steps = 0
        flag = False
        count_of_dones = 0

        for epoch in range(epochs):
            done = False
            state = torch.Tensor([self.env.reset()])
            entropies = []
            log_probs = []
            rewards = []
            ep_reward = 0
            while not done:
                action, log_prob, entropy = self.agent.select_action(state)
                action = action.cpu()
                action = softmax(action.numpy()[0])
                next_state, reward, done, _ = self.env.step(action)

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                ep_reward += reward
                state = torch.Tensor([next_state])
                
                total_steps += 1
                if done:
                    count_of_dones += 1
                    flag = True
            
                if flag and count_of_dones % print_every == 0:
                        print(f'Score at timestep {total_steps}: {ep_reward}.')
                        flag = False

                if total_steps >= timesteps:
                    break

            self.agent.update_parameters(
                rewards, log_probs, entropies, self.gamma)
            
        self.env.close()

    def learn(self, timesteps, print_every=100):
        """
        Trains the agent

        Params
        ======
            timesteps (int): Number of timesteps the agent should interact with the environment
            print_every (int): Verbosity control
        """
        self.agent.model.train()
        self.train(timesteps, print_every)

    def predict(self, state):
        """Returns agent's action based on a given state

        Args:
            state (array_like): Current environment state

        Returns:
            action (array_like): Agent's action
        """        
        self.agent.model.eval()
        state = torch.from_numpy(state).float()
        action, log_prob, entropy = self.agent.select_action(state)
        normalized_action = softmax(action.cpu().numpy())
        return normalized_action
        # return action

    def save(self, file_name):
        """
        Saves trained model

        Params
        =====
        filepath(str) : folder path to save the agent
        """
        torch.save(self.agent.model.state_dict(), file_name)

    def load(self, file_name):
        """
        Loads trained model

        Params
        =====
        filepath(str) : folder path to save the agent
        """
        self.model.load_state_dict(torch.load(file_name))
