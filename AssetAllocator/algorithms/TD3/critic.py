import torch
import torch.nn as nn
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lr = 0.1):
        """
        state dim -> Num of states
        actor dim -> Number of actions
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
        x = torch.cat([state, action], dim = 1)
        out = self.linear_relu_stack(x)
        return out