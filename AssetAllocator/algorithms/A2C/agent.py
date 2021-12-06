from .a2c import Actor, Critic, A2CLearner, Runner
import torch

class A2CAgent: 
    
    """This is the agent class for the A2C Agent.

    Original paper can be found at https://arxiv.org/abs/1802.09477

    This implementation was adapted from https://github.com/saashanair/rl-series/tree/master/td3
    
    """
    def __init__(self, env, hidden_dim = 256, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        
        """Initializes the A2C Agent

        Args:
            env ([type]): Gym environment for the agent to interact with
            hidden_dim (int, optional): Size of hidden layer neurons. Defaults to 256.
            device (str, optional): One of cuda or cpu. Defaults to 'cuda'.
            memory_dim ([type], optional): Size of replay buffer. Defaults to 100_000.
            actor_lr ([type], optional): Actor's learning rate. Defaults to 1e-3.
            critic_lr ([type], optional): Critic's learning rate. Defaults to 1e-3
        """  
        
        self.env = env
           
        n_actions = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        
        actor = Actor(state_dim, hidden_dim, n_actions)
        critic = Critic(state_dim, hidden_dim)

        self.learner = A2CLearner(actor, critic, gamma, entropy_beta,
                             actor_lr, critic_lr, max_grad_norm)
        self.runner = Runner(env, actor)
        
    def learn(self, timesteps, print_every = 1000):
        """
        Trains the agent
        Params
        ======
            timesteps (int): Number of timesteps the agent should interact with the environment
            print_every (int): Verbosity control
        """
        total_steps = timesteps//self.env.episode_length + 1

        while self.runner.steps <= timesteps:
            memory = self.runner.run(total_steps, print_every)
            self.learner.learn(memory, self.runner.steps, discount_rewards=True)
            
    def predict(self, state):

        """Returns agent's action based on a given state
        Args:
            state (array_like): Current environment state
        Returns:
            action (array_like): Agent's action
        """         
        return self.learner.predict(state)
    
    def save(self, file_name):
        """
        Saves trained model
        Params
        =====
        filepath(str) : folder path to save the agent
        """
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.learner.actor_optim.state_dict(), filename + '_actor_optimizer')
        
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.learner.critic_optim.state_dict(), filename + '_critic_optimizer')
        
    def load(self, file_name):
        """
        Loads trained model
        Params
        =====
        filepath(str) : folder path to save the agent
        """
        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.learner.actor_optim.load_state_dict(torch.load(filename + '_actor_optimizer'))
        
        self.critic.load_state_dict(torch.load(filename + '_critic'))
        self.learner.critic_optim.load_state_dict(torch.load(filename + '_critic_optimizer'))
        
