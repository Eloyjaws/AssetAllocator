import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lookback_dim, add_lstm = True, num_layers = 3,
                 lr = 0.1, max_action = 1, dropout = 0.2):
        """
        state dim -> Num of states
        actor dim -> Number of actions
        """
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.max_action = max_action
        
        in_dim = state_dim//(action_dim - 1)
        
        if add_lstm:
            self.lstm = nn.LSTM(action_dim - 1, state_dim//2, num_layers = num_layers, batch_first = True, 
                            dropout = dropout, bidirectional = True)
        
            self.linear_relu_stack = nn.Sequential(
                    nn.Linear(state_dim * lookback_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
            )
        else:
            self.lstm = None
            self.linear_relu_stack =  self.linear_relu_stack = nn.Sequential(
                                        nn.Linear(state_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim),
                                )

        self.optimizer = optim.Adam(self.linear_relu_stack.parameters(), 
                                    lr = lr)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 2)

    def forward(self, state):
        if self.lstm:
            state = state.reshape(state.shape[0], -1, self.action_dim - 1)
            out, _ = self.lstm(state)
            out = self.linear_relu_stack(out.reshape(state.shape[0],-1)) 
        else:
            out = self.linear_relu_stack(state)
            
        out = nn.Softmax(dim = 1)(out)
        return out * self.max_action