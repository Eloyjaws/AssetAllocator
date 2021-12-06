from .replay_buffer import ReplayBuffer 
from .network import NAFNetwork
from .noise import OUNoise
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np 
import torch.optim as optim
import random
import copy

class NAFAgent():
    """This is the agent class for the Normalized Advantage Function.

    Original paper can be found at https://arxiv.org/abs/1906.04594

    This implementation was adapted from https://github.com/BY571/Normalized-Advantage-Function-NAF-
    
    """

    def __init__(self,
                 env,
                 device = 'cuda',
                 layer_size = 256,
                 BATCH_SIZE = 128,
                 BUFFER_SIZE = 10_000,
                 LR = 1e-3,
                 TAU = 1e-3,
                 GAMMA = 0.99,
                 UPDATE_EVERY = 2,
                 NUPDATES = 1,
                 seed = 0):
        """Initialize an NAFAgent object.
        
        Params
        ======
            env (PortfolioGymEnv): instance of environment
            device: device type (one of cuda or cpu)
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.env = env
        
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.observation_space.shape[0]
        
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.NUPDATES = NUPDATES
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0


        self.action_step = 4
        self.last_action = None

        # Q-Network
        self.qnetwork_local = NAFNetwork(self.state_size, self.action_size,layer_size, seed, self.device).to(device)
        self.qnetwork_target = NAFNetwork(self.state_size, self.action_size,layer_size, seed, self.device).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Noise process
        self.noise = OUNoise(self.action_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        """
        Trains the agent

        Params
        =====
        state (array_like): current state
        action (array_like): current action
        reward (array_like): reward for current state and action pair
        next_state (array_like): next state
        done(array_like): current end status
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                Q_losses = []
                for _ in range(self.NUPDATES):
                    experiences = self.memory.sample()
                    loss = self._learn(experiences)
                    self.Q_updates += 1
                    Q_losses.append(loss)

    def predict(self, state):
        """Returns the action for a given state
        
        Params
        ======
            state (array_like): current state
            
        """

        state = torch.from_numpy(state).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action, _, _ = self.qnetwork_local(state.unsqueeze(0))
            action = torch.nn.Softmax(dim = 1)(action)
            
        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy()



    def _learn(self, experiences):
        """
        Helper method to update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA * V_ * (1 - dones))
        
        # Get expected Q values from local model
        _, Q, _ = self.qnetwork_local(states, actions)

        # Compute loss
        loss = F.mse_loss(Q, V_targets)
        
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        self.noise.reset()
        
        return loss.detach().cpu().numpy()  
    
    def learn(self, timesteps, print_every = 1):
        """
        Trains the agent

        Params
        ======
            timesteps (int): Number of timesteps the agent should interact with the environment
            print_every (int): Verbosity control
        """

        epochs = timesteps//self.env.episode_length + 1
        
        timestep = 0
        count_of_dones = 0
        flag = False
        
        for _ in range(epochs):
            done = False
            state = self.env.reset()
    
            self.score = 0
            while not done and timestep < timesteps:
                # generate noisy action
                action = self.predict(state)

                # execute the action in the environment
                next_state, reward, done, _ = self.env.step(np.array(action))

                # update the networks
                self.step(state, action, reward, next_state, done)


                #get the next state
                state = next_state

                self.score += reward
                timestep += 1
                
                if done:
                    count_of_dones += 1
                    flag = True
                    
                if flag and count_of_dones % print_every == 0:
                    print(f'Score at timestep {timestep}: {self.score}.')
                    flag = False
        
        #print(f'Final score is {self.score} after {timesteps} timesteps.')

    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def save(self, filepath):
        """
        Saves trained model

        Params
        =====
        filepath(str) : folder path to save the agent
        """
        torch.save(self.qnetwork_target.state_dict(),filepath + "_target")
        torch.save(self.qnetwork_local.state_dict(), filepath + "_local")
        torch.save(self.optimizer.state_dict(), filepath + "_optimizer")

    def load(self, filepath):
        """
        Load trained model

        Params
        =====
        filepath(str) : folder path to save the agent
        """
        self.qnetwork_local.load_state_dict(torch.load(filepath + '_local'))
        self.qnetwork_target.load_state_dict(torch.load(filepath + '_target')) 
        self.optimizer.load_state_dict(torch.load(filepath + '_optimizer'))