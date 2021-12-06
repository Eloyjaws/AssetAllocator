import torch
import torch.nn as nn 
from torch.distributions import MultivariateNormal

class NAFNetwork(nn.Module):
    """
        Neural Network Approximator for NAF.

        Original paper can be found at https://arxiv.org/abs/1906.04594

        This implementation was adapted from https://github.com/BY571/Normalized-Advantage-Function-NAF-
    """

    def __init__(self, state_size, action_size,layer_size, seed, device):
        """
        Computes the forward pass of the NAF Network

        Params
        =====
        state_size: state space size
        action_size: action space size
        layer_size: number of neurons in hidden layer
        seef: random seed
        device: one of cuda or cpu
        """
        super(NAFNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.device = device
                
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
        
    def forward(self, input_, action=None):
        """
        Computes the forward pass of the NAF Network

        Params
        =====
        input_ : State tensor
        action : Action tensor
        """
        x = torch.relu(self.head_1(input_))
        x = torch.relu(self.ff_1(x))
        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = action_value.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(self.device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V
        
        
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        #dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        #action = action_value.squeeze(-1)
        
        return action, Q, V
    