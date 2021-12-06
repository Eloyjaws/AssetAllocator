import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Network import Actor, Critic
from Replay_Memory import ReplayMemory
from OU_Noise import OrnsteinUhlenbeckNoise

class DDPGAgent:
    """This is the agent class for the DDPG Agent.

    Original paper can be found at https://arxiv.org/abs/1509.02971

    This implementation was adapted from https://github.com/saashanair/rl-series/tree/master/ddpg
    
    """

    def __init__(
        self, 
        state_dim, 
        action_dim,
        max_action, 
        device, 
        memory_capacity=10000, 
        discount=0.99, 
        tau=0.005, 
        sigma=0.2, 
        theta=0.15, 
        actor_lr=1e-4, 
        critic_lr=1e-3, 
        train_mode=True):

        """Initializes the DDPG Agent

        Args:
            state_dim (int): State space dimension
            action_dim (int): Action space dimension
            max_action (int): the max value of the range in the action space (assumes a symmetric range in the action space)
            device (str, optional): One of cuda or cpu. Defaults to 'cuda'.
            memory_capacity ([type], optional): Size of replay buffer. Defaults to 10_000.
            discount (float, optional): Reward discount factor. Defaults to 0.99.
            tau (float, optional): Polyak averaging soft updates factor (i.e., soft updating of the target networks). Defaults to 0.005.
            sigma (float, optional): Amount of noise to be applied to the OU process. Defaults to 0.2.
            theta (float, optional): Amount of frictional force to be applied in OU noise generation. Defaults to 0.15.
            actor_lr ([type], optional): Actor's learning rate. Defaults to 1e-4.
            critic_lr ([type], optional): Critic's learning rate. Defaults to 1e-3.
            train_mode (bool, optional): Training or eval mode flag. Defaults to True.
        """      

        self.train_mode = train_mode # whether the agent is in training or testing mode

        self.state_dim = state_dim # dimension of the state space
        self.action_dim = action_dim # dimension of the action space
        
        self.device = device # defines which cuda or cpu device is to be used to run the networks
        self.discount = discount # denoted a gamma in the equation for computation of the Q-value
        self.tau = tau # defines the factor used for Polyak averaging (i.e., soft updating of the target networks)
        self.max_action = max_action # the max value of the range in the action space (assumes a symmetric range in the action space)
        
        # create an instance of the replay buffer
        self.memory = ReplayMemory(memory_capacity)

        # create an instance of the noise generating process
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma, theta=theta)

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

        # for test mode
        if not self.train_mode:
            self.actor.eval()
            self.critic.eval()
            self.ounoise = None

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

    def select_action(self, state):
        """
        Function to return the appropriate action for the given state.
        During training, it adds a zero-mean OU noise to the action to encourage exploration.
        During testing, no noise is added to the action decision.
        Parameters
        ---
        state (array_like): The current state of the environment as observed by the agent
        
        Returns:
            action: A numpy array representing the noisy action to be performed by the agent in the current state
        """

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        
        self.actor.eval()
        act = self.actor(state).cpu().data.numpy().flatten() # performs inference using the actor based on the current state as the input and returns the corresponding np array
        self.actor.train()

        noise = 0.0

        ## for adding Gaussian noise (to use, update the code pass the exploration noise as input)
        #if self.train_mode:
        #	noise = np.random.normal(0.0, exploration_noise, size=act.shape) # generate the zero-mean gaussian noise with standard deviation determined by exploration_noise

        # for adding OU noise
        if self.train_mode:
            noise = self.ou_noise.generate_noise()

        noisy_action = act + noise
        noisy_action = noisy_action.clip(min=-self.max_action, max=self.max_action) # to ensure that the noisy action being returned is within the limit of "legal" actions afforded to the agent; assumes action range is symmetric

        return noisy_action

    def learn(self, batchsize):
        """
        Function to perform the updates on the 4 neural networks that run the DDPG algorithm.
        Args: 
            batchsize (int): Number of experiences to be randomly sampled from the memory for the agent to learn from
        """

        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device) # a batch of experiences randomly sampled form the memory

        # ensure that the actions and rewards tensors have the appropriate shapes
        actions = actions.view(-1, self.action_dim) 
        rewards = rewards.view(-1, 1)

        with torch.no_grad():
            # generate target actions
            target_action = self.target_actor(next_states)

            # calculate TD-Target
            target_q = self.target_critic(next_states, target_action)
            target_q[dones] = 0.0 # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
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


    def soft_update_net(self, source_net_params, target_net_params):
        """
        Perform Polyak averaging to update the parameters of the provided network
        Args:
            source_net_params (list): trainable parameters of the source, ie. current version of the network
            target_net_params (list): trainable parameters of the corresponding target network
        """

        for source_param, target_param in zip(source_net_params, target_net_params):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def soft_update_targets(self):
        """ Function that calls Polyak averaging on both target networks """

        self.soft_update_net(self.actor.parameters(), self.target_actor.parameters())
        self.soft_update_net(self.critic.parameters(), self.target_critic.parameters())

    def save(self, path, model_name):
        """
            Saves trained model

            Params
            =====
            path(str) : folder path to save the agent's weights
            name(str) : name to save the agent's weights 
        """
        self.actor.save_model('{}/{}_actor'.format(path, model_name))
        self.critic.save_model('{}/{}_critic'.format(path, model_name))

    def load(self, path, model_name):
        """
            Loads trained model

            Params
            =====
            path(str) : folder path to the agent's weights
            name(str) : name of the saved agent's weights 
        """
        self.actor.load_model('{}/{}_actor'.format(path, model_name))
        self.critic.load_model('{}/{}_critic'.format(path, model_name))