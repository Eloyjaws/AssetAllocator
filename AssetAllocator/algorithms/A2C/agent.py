from .a2c import Actor, Critic, A2CLearner, Runner
import torch

class A2CAgent:
    def __init__(self, env, hidden_dim = 256, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        
        self.env = env
           
        n_actions = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        
        actor = Actor(state_dim, hidden_dim, n_actions)
        critic = Critic(state_dim, hidden_dim)

        self.learner = A2CLearner(actor, critic, gamma, entropy_beta,
                             actor_lr, critic_lr, max_grad_norm)
        self.runner = Runner(env, actor)
        
    def learn(self, timesteps, print_every = 1000):
        total_steps = timesteps//self.env.episode_length + 1

        while self.runner.steps <= timesteps:
            memory = self.runner.run(total_steps, print_every)
            self.learner.learn(memory, self.runner.steps, discount_rewards=True)
            
    def predict(self, state):
        return self.learner.predict(state)
    
    def save(self, file_name):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.learner.actor_optim.state_dict(), filename + '_actor_optimizer')
        
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.learner.critic_optim.state_dict(), filename + '_critic_optimizer')
        
    def load(self, file_name):
        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.learner.actor_optim.load_state_dict(torch.load(filename + '_actor_optimizer'))
        
        self.critic.load_state_dict(torch.load(filename + '_critic'))
        self.learner.critic_optim.load_state_dict(torch.load(filename + '_critic_optimizer'))
        
