"""
Script that contains the training and testing loops
"""
import gym
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Network import Actor, Critic
from .Replay_Memory import ReplayMemory
from .OU_Noise import OrnsteinUhlenbeckNoise


class DDPGAgentHelper:

    def __init__(self, env, state_dim, action_dim, max_action, device, memory_capacity=10000, num_memory_fill_episodes=10, discount=0.99, tau=0.005, sigma=0.2, theta=0.15, actor_lr=1e-4, critic_lr=1e-3, batch_size=64, warmup_steps = 100):
        self.env = env
        self.batch_size = batch_size

        self.state_dim = state_dim  # dimension of the state space
        self.action_dim = action_dim  # dimension of the action space

        self.device = device  # defines which cuda or cpu device is to be used to run the networks
        # denoted a gamma in the equation for computation of the Q-value
        self.discount = discount
        # defines the factor used for Polyak averaging (i.e., soft updating of the target networks)
        self.tau = tau
        # the max value of the range in the action space (assumes a symmetric range in the action space)
        self.max_action = max_action
        self.warmup_steps = warmup_steps
        # create an instance of the replay buffer
        self.memory_capacity = memory_capacity
        self.num_memory_fill_episodes = num_memory_fill_episodes
        self.memory = ReplayMemory(memory_capacity)

        # create an instance of the noise generating process
        self.ou_noise = OrnsteinUhlenbeckNoise(
            mu=np.zeros(self.action_dim), sigma=sigma, theta=theta)

        # instances of the networks for the actor and the critic
        self.actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.critic = Critic(state_dim, action_dim, critic_lr)

        # instance of the target networks for the actor and the critic
        self.target_actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.target_critic = Critic(state_dim, action_dim, critic_lr)

        # initialise the targets to the same weight as their corresponding current networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # since we do not learn/train on the target networks
        self.target_actor.eval()
        self.target_critic.eval()

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

    def fill_memory(self):
        epochs = self.warmup_steps//self.env.episode_length + 1
        for epoch in range(epochs):
            state = self.env.reset()
            done = False

            while not done:
                action = self.env.action_space.sample()  # do random action for warmup
                action = action/action.sum() #normalize random actions
                next_state, reward, done, _ = self.env.step(action)
                # store the transition to memory
                self.memory.store([state, action, next_state, reward, done])
                state = next_state
        print("Done filling memory")

    @staticmethod
    def _softmax(x, axis=0):
        # Use the LogSumExp Trick
        max_val = np.amax(x, axis=axis, keepdims=True)
        x = x - max_val

        # Softmax
        num = np.exp(x)
        denum = num.sum(axis=axis, keepdims=True)
        softmax = num/denum
        return softmax

    def select_action(self, state):
        """
        Function to return the appropriate action for the given state.
        During training, it adds a zero-mean OU noise to the action to encourage exploration.
        During testing, no noise is added to the action decision.
        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent

        Returns
        ---
        A numpy array representing the noisy action to be performed by the agent in the current state
        """

        if not torch.is_tensor(state):
            state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)

        self.actor.eval()
        # performs inference using the actor based on the current state as the input and returns the corresponding np array
        act = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        noise = 0.0

        # for adding Gaussian noise (to use, update the code pass the exploration noise as input)
        # if self.train_mode:
        #	noise = np.random.normal(0.0, exploration_noise, size=act.shape) # generate the zero-mean gaussian noise with standard deviation determined by exploration_noise

        # for adding OU noise
        # if self.train_mode:
        noise = self.ou_noise.generate_noise()
        noisy_action = act + noise
        # to ensure that the noisy action being returned is within the limit of "legal" actions afforded to the agent;
        noisy_action = noisy_action.clip(
            min=0, max=self.max_action)
        return DDPGAgentHelper._softmax(noisy_action)

    def _learn(self, current_step):
        """
        Function to perform the updates on the 4 neural networks that run the DDPG algorithm.
        Parameters
        ---
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from
        Returns
        ---
        none
        """
        if len(self.memory) < self.batch_size:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size, self.device)  # a batch of experiences randomly sampled form the memory

        # ensure that the actions and rewards tensors have the appropriate shapes
        actions = actions.view(-1, self.action_dim)
        rewards = rewards.view(-1, 1)

        with torch.no_grad():
            # generate target actions
            target_action = self.target_actor(next_states)

            # calculate TD-Target
            target_q = self.target_critic(next_states, target_action)
            # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
            target_q[dones] = 0.0
            y = rewards + self.discount * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y).mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # actor loss is calculated by a gradient ascent along the crtic, thus need to apply the negative sign to convert to a gradient descent
        pred_current_actions = self.actor(states)
        pred_current_q = self.critic(states, pred_current_actions)
        actor_loss = - pred_current_q.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # apply slow-update to the target networks
        self.soft_update_targets()

    def learn(self, timesteps, print_every=100):
        self.fill_memory()  # to populate the replay buffer before learning begins
        self.train(timesteps, print_every)

    def predict(self, state):
        ou = self.ou_noise
        self.actor.eval()
        self.critic.eval()
        self.ounoise = None

        action = self.select_action(state)

        self.actor.train()
        self.critic.train()
        self.ounoise = ou
        return action

    def soft_update_net(self, source_net_params, target_net_params):
        """
        Function to perform Polyak averaging to update the parameters of the provided network
        Parameters
        ---
        source_net_params: list
            trainable parameters of the source, ie. current version of the network
        target_net_params: list
            trainable parameters of the corresponding target network
        Returns
        ---
        none
        """

        for source_param, target_param in zip(source_net_params, target_net_params):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def soft_update_targets(self):
        """ Function that calls Polyak averaging on all three target networks """

        self.soft_update_net(self.actor.parameters(),
                             self.target_actor.parameters())
        self.soft_update_net(self.critic.parameters(),
                             self.target_critic.parameters())

    def train(self, timesteps, print_every):
        reward_history = []  # tracks the reward per episode
        best_score = -np.inf

        epochs = timesteps//self.env.episode_length + 1
        total_steps = 0
        flag = False
        count_of_dones = 0
        
        for ep_cnt in range(epochs):
            done = False
            state = self.env.reset()
            ep_reward = 0

            while not done:
                action = self.select_action(state)  # generate noisy action
                # print("Action:", action)
                next_state, reward, done, _ = self.env.step(
                    action)  # execute the action in the environment
                # store the interaction in the replay buffer
                self.memory.store([state, action, next_state, reward, done])

                self._learn(total_steps)  # update the networks

                state = next_state
                total_steps += 1
                ep_reward += reward
                
                if done:
                    count_of_dones += 1
                    flag = True
            
                if flag and count_of_dones % print_every == 0:
                        print(f'Score at timestep {total_steps}: {ep_reward}.')
                        flag = False

                if total_steps >= timesteps:
                    break
                
    def save(self, file_name):
        self.actor.save_model(f"{file_name}_actor")
        self.critic.save_model(f"{file_name}_critic")

    def load(self, file_name):
        self.actor.load_model(f"{file_name}_actor")
        self.critic.load_model(f"{file_name}_critic")

    # def save(self, path, model_name):
    #     self.actor.save_model('{}/{}_actor'.format(path, model_name))
    #     self.critic.save_model('{}/{}_critic'.format(path, model_name))

    # def load(self, path, model_name):
    #     self.actor.load_model('{}/{}_actor'.format(path, model_name))
    #     self.critic.load_model('{}/{}_critic'.format(path, model_name))


default_device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def DDPGAgent(env, device=default_device):
    ddpg_agent = DDPGAgentHelper(env=env,
                           state_dim=env.observation_space.shape[0],
                           action_dim=env.action_space.shape[0],
                           max_action=env.action_space.high[0],
                           device=device,
                           memory_capacity=10000,
                           discount=0.99,
                           tau=0.005,
                           sigma=0.2,
                           theta=0.15,
                           actor_lr=1e-4,
                           critic_lr=1e-3,
                           batch_size=64)
    return ddpg_agent


