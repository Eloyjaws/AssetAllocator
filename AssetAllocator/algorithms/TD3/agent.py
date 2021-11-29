import torch
import torch.nn as nn
import torch.nn.functional as F
from .actor import Actor
from .critic import Critic
from .memory import Memory
import numpy as np

class TD3Agent:
    def __init__(
            self,
            env,
            hidden_dim = 256,
            device = 'cuda',
            memory_dim=100_000,
            max_action = 1,
            discount=0.99,
            update_freq=2,
            tau=0.005,
            policy_noise_std=0.2,
            policy_noise_clip=0.5,
            actor_lr= 1e-3,
            critic_lr= 1e-3,
            batch_size=128,
            exploration_noise=0.1,
            num_layers = 3,
            dropout = 0.2,
            add_lstm = False,
            warmup_steps = 100):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self.action_dim = env.action_space.shape[-1]
        self.lookback = env.lookback_period
        self.device = device
        self.max_action = max_action
        self.memory_dim = memory_dim
        self.discount = discount
        self.update_freq = update_freq
        self.tau = tau
        self.policy_noise_std = policy_noise_std
        self.policy_noise_clip = policy_noise_clip
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.eval = False
        self.num_layers = 3
        self.dropout = 0.2
        self.warmup_steps = warmup_steps

        # Instatiate Memory Buffer
        self.memory = Memory(self.memory_dim)

        # Instantiate Actor and Target Actor
        self.actor = Actor(state_dim = self.state_dim, 
                           action_dim = self.action_dim, 
                           hidden_dim = self.hidden_dim, 
                           lookback_dim = self.lookback, 
                           num_layers = self.num_layers,
                           lr = self.actor_lr, 
                           max_action = self.max_action, 
                           dropout = self.dropout, 
                           add_lstm = add_lstm)       
        self.actor.to(self.device)
        
        self.target_actor = Actor(state_dim = self.state_dim, 
                           action_dim = self.action_dim, 
                           hidden_dim = self.hidden_dim, 
                           lookback_dim = self.lookback, 
                           num_layers = self.num_layers,
                           lr = self.actor_lr, 
                           max_action = self.max_action, 
                           dropout = self.dropout,
                           add_lstm = add_lstm)       
        self.target_actor.to(self.device)

        # Instantiate Critic 1 and Target Critic 1
        self.critic1 = Critic(state_dim = self.state_dim, 
                              action_dim = self.action_dim , 
                              hidden_dim = self.hidden_dim,
                              lr = self.critic_lr)
        self.critic1.to(self.device)
        
        self.target_critic1 = Critic(state_dim = self.state_dim, 
                              action_dim = self.action_dim , 
                              hidden_dim = self.hidden_dim,
                              lr = self.critic_lr)
        self.target_critic1.to(self.device)

        # Instantiate Critic 2 and Target Critic 2
        self.critic2 = Critic(state_dim = self.state_dim, 
                              action_dim = self.action_dim , 
                              hidden_dim = self.hidden_dim,
                              lr = self.critic_lr)
        self.critic2.to(self.device)
        
        self.target_critic2 = Critic(state_dim = self.state_dim, 
                              action_dim = self.action_dim , 
                              hidden_dim = self.hidden_dim,
                              lr = self.critic_lr)
        self.target_critic2.to(self.device)

        # Copy weight to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # since we do not learn/train on the target networks
        self.target_actor.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()

    # for test mode
    def eval_mode(self):
        self.eval = True
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def select_action(self, state, exploration_noise=0.1):

        if not torch.is_tensor(state):
            state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)

        # Forward pass through actor network
        action = self.actor(state).cpu().data.numpy().flatten()

        if self.eval:
            exploration_noise = 0.0

        noise = np.random.normal(0.0, exploration_noise, size=action.shape)

        noisy_action = (action + noise)
        noisy_action = TD3Agent._softmax(noisy_action)

        return noisy_action

    def _update(self, source_net_params, target_net_params):
        for source_param, target_param in zip(
                source_net_params, target_net_params):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def update_targets(self):
        self._update(self.actor.parameters(), self.target_actor.parameters())
        self._update(
            self.critic1.parameters(),
            self.target_critic1.parameters())
        self._update(
            self.critic2.parameters(),
            self.target_critic2.parameters())
    
    @staticmethod
    def _softmax(x, axis = 0):
        # Use the LogSumExp Trick
        max_val = np.amax(x, axis=axis, keepdims = True)
        x = x - max_val

        # Softmax
        num = np.exp(x)
        denum = num.sum(axis = axis, keepdims = True)
        softmax = num/denum
        return softmax

    def _learn(self, iteration):
        if len(self.memory) < self.batch_size:
            return

    	# Memory Replay
        states, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size, self.device)

        actions =  nn.Softmax(dim = 1)(actions)
        actions = actions.view(-1, self.action_dim)
        rewards = rewards.view(-1, 1)
        
        #print(states.shape, actions.shape, next_states.shape, rewards.shape, dones.shape)
        
        with torch.no_grad():
            # Target Policy Smoothing
            pred_action = self.target_actor(next_states)
            noise = torch.zeros_like(pred_action).normal_(0, self.policy_noise_std).to(self.device)
            noisy_pred_action = pred_action + noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)
            noisy_pred_action = nn.Softmax(dim = 1)(noisy_pred_action)

            # Clipped Double Q Learning
            target_q1 = self.target_critic1(next_states, noisy_pred_action)
            target_q2 = self.target_critic2(next_states, noisy_pred_action)
            target_q = torch.min(target_q1, target_q2).detach()
            target_q[dones] = 0.0
            y = rewards + self.discount * target_q

        # Loss Computation
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, y).mean()
        critic2_loss = F.mse_loss(current_q2, y).mean()

        # Gradient Descent on critics
        self.critic1.optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2.optimizer.step()


        # delayed policy and target updates
        if iteration % self.update_freq == 0:

            # Compute actor loss
            # print(states.shape)
            pred_current_actions = self.actor(states)
            pred_current_q1 = self.critic1(states, pred_current_actions)
            actor_loss = - pred_current_q1.mean()
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # apply slow-update to all three target networks
            self.update_targets()

    def fill_memory(self):
        fill_memory_epochs = self.warmup_steps//self.env.episode_length
        
        for _ in range(fill_memory_epochs):
            state = self.env.reset()
            done = False

            while not done:
                # do random action for warmup
                action = self.env.action_space.sample() 
                action = np.array(action/sum(action))
                next_state, reward, done, _ = self.env.step(action)
                
                # store the transition to memory
                self.memory.store([state, action, next_state, reward, done]) 
                state = next_state

    def train(self, total_steps, timesteps, print_every, count_of_dones):
        done = False
        state = self.env.reset()
        ep_reward = 0
        flag = False
        
        while not done:
            action = self.select_action(state, self.exploration_noise)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.store([state, action, next_state, reward, done])
            self._learn(total_steps)
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
        
        return total_steps, count_of_dones

    def predict(self, state):
        self.eval_mode()
        action = self.select_action(state, self.exploration_noise)
        return action

    def learn(self, timesteps, print_every = 1):
        epochs = timesteps//self.env.episode_length + 1
        self.fill_memory()
        print('Startup memory filled!')
        
        count_of_dones = 0
        
        total_steps = 0
        for _ in range(epochs):
            total_steps, count_of_dones = self.train(total_steps, timesteps, print_every, count_of_dones)

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + '_critic1')
        torch.save(self.critic1.optimizer.state_dict(), filename + '_critic1_optimizer')

        torch.save(self.critic2.state_dict(), filename + '_critic2')
        torch.save(self.critic2.optimizer.state_dict(), filename + '_critic2_optimizer')

        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor.optimizer.state_dict(), filename + '_actor_optimizer')

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + '_critic1'))
        self.critic1.optimizer.load_state_dict(torch.load(filename + '_critic1_optimizer'))

        self.critic2.load_state_dict(torch.load(filename + '_critic2'))
        self.critic2.optimizer.load_state_dict(torch.load(filename + '_critic2_optimizer'))

        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.actor.optimizer.load_state_dict(torch.load(filename + '_actor_optimizer'))
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
