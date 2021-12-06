import copy
import numpy as np

class OUNoise:
    """
    Implements the Ornstein-Uhlenbeck Noise
    
    Original paper can be found at https://arxiv.org/abs/1906.04594

    This implementation was adapted from https://github.com/BY571/Normalized-Advantage-Function-NAF-  
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        
        Params
        =======
        size: state space size
        seed: random seed
        mu: mean value of the OUNoise
        theta: theta value of the OUNoise
        sigma: standard deviation of the OUNoise
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
