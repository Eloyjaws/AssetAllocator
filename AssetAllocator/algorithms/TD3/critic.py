import torch
import torch.nn as nn
import torch.optim as optim

class Critic(nn.Module):
    """This is the critic network for the TD3 Agent.

    Original paper can be found at https://arxiv.org/abs/1802.09477

    This implementation was adapted from https://github.com/saashanair/rl-series/tree/master/td3
    
    """
    def __init__(self, state_dim, action_dim, hidden_dim, lr = 0.1):
        """Initializes the TD3 Critic Network

        Args:
            state_dim (int): State space dimension
            action_dim (int): Action space dimension
            hidden_dim (int): Size of hidden layer
            lr (float, optional): Learning rate. Defaults to 0.1.
        """        
        super(Critic, self).__init__()

        self.linear_relu_stack = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
        )

        self.optimizer = optim.Adam(self.linear_relu_stack.parameters(), 
                                    lr = lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 2)
        
    def forward(self, state, action):
        """Forward pass

        Args:
            state (array_like): Current environment state
            action (array_like): Current agent's action

        Returns:
            out: State-Action Values
        """
        x = torch.cat([state, action], dim = 1)
        out = self.linear_relu_stack(x)
        return out