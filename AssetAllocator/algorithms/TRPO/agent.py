
from itertools import count
import signal
import sys
import os
import time
sys.path.append('..')
import numpy as np
import gym
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import scipy.optimize
import matplotlib.pyplot as plt
from .value import Value
from .policy import Policy
from .utils import *
from .trpo import trpo_step

class TRPOAgent:
    """
        This is the agent class for the Trust Region Policy Optmization Algorithm.

        Original paper can be found at https://arxiv.org/abs/1502.05477

        This implementation was adapted from https://github.com/MEfeTiryaki/trpo/blob/master/train.py    
    """
    def __init__(
        self,
        env,
        device = 'cuda',
        damping=0.1,        
        episode_length= 2000,
        fisher_ratio=1,
        gamma=0.995,
        l2_reg=0.001,
        lambda_=0.97,
        lr=0.001,
        max_iteration_number=200,
        max_kl=0.01,
        save=False,
        seed=543,
        val_opt_iter=200,
        value_memory=1,
        value_memory_shuffle=False):
        """Initialize a TRPOAgent object.
        
        Params
        ======
            env (PortfolioGymEnv): instance of environment
            device: device type (one of cuda or cpu)
            damping: policy optimization parameter
            episode_length: max step size for one episode
            fisher_ratio: policy optimization parameter
            l2_reg: l2 regularization regression
            lambda_: gae
            max_iteration_number: max policy iteration number
            max_kl: max kl divergence (policy optimization parameter)
            val_opt_iter: iteration number for value function learning
            lr (float): learning rate
            gamma (float): discount factor
            seed (int): random seed
        """        
        self.env = env
        self.device = device
        self.damping = damping
        self.episode_length = episode_length
        self.fisher_ratio = fisher_ratio
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.lambda_ = lambda_
        self.lr = lr        
        self.max_iteration_number = max_iteration_number
        self.max_kl = max_kl
        self.val_opt_iter = val_opt_iter
        self.save = save
        self.seed = seed
        self.value_memory = value_memory
        self.value_memory_shuffle = value_memory_shuffle
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[-1]
        
        self.policy_net = Policy(self.state_dim, self.action_dim, 256, device).to(device)
        self.value_net = Value(self.state_dim, 256).to(device)
        
        
    def signal_handler(self, sig, frame):
        """ 
            Signal Handler to save the networks when shutting down via ctrl+C
            Parameters:
            ==========
            sig
            frame
            
            Returns:
        """
        if(self.save):
            valueParam = get_flat_params_from(self.value_net)
            policyParam = get_flat_params_from(self.policy_net)
            saveParameterCsv(valueParam,self.load_dir+"/ValueNet")
            saveParameterCsv(policyParam,self.load_dir+"/PolicyNet")
            print("Networks are saved in "+self.load_dir+"/")

        print('Closing!!')
        env.close()
        sys.exit(0)

    def prepare_data(self, batch, valueBatch, previousBatch):
        """ 
            Get the batch data and calculate value,return and generalized advantage
            Parameters:
            ==========
            batch
            valueBatch
            previousBatch
            
        """

        stateList = [ torch.from_numpy(np.concatenate(x,axis=0)).to(self.device) for x in batch["states"]]
        actionsList = [torch.from_numpy(np.concatenate(x,axis=0)).to(self.device) for x in batch["actions"]]

        for states in stateList:
            value = self.value_net.forward(states)
            batch["values"].append(value)

        advantagesList = []
        returnsList = []
        rewardsList = []
        for rewards,values,masks in zip(batch["rewards"],batch["values"],batch["mask"]):
            returns = torch.Tensor(len(rewards),1).to(self.device)
            advantages = torch.Tensor(len(rewards),1).to(self.device)
            deltas = torch.Tensor(len(rewards),1).to(self.device)

            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(len(rewards))):
                returns[i] = rewards[i] + self.gamma * prev_value * masks[i] # TD
                # returns[i] = rewards[i] + self.gamma * prev_return * masks[i] # Monte Carlo
                deltas[i] = rewards[i] + self.gamma * prev_value * masks[i]- values.data[i]
                advantages[i] = deltas[i] + self.gamma * self.lambda_* prev_advantage* masks[i]

                prev_return = returns[i, 0]
                prev_value = values.data[i, 0]
                prev_advantage = advantages[i, 0]
            returnsList.append(returns)
            advantagesList.append(advantages)
            rewardsList.append(torch.Tensor(rewards).to(self.device))


        batch["states"] = torch.cat(stateList,0)
        batch["actions"] = torch.cat(actionsList,0)
        batch["rewards"] = torch.cat(rewardsList,0)
        batch["returns"] = torch.cat(returnsList,0)

        advantagesList = torch.cat(advantagesList,0)
        batch["advantages"] = (advantagesList- advantagesList.mean()) / advantagesList.std()

        valueBatch["states"] = torch.cat(( previousBatch["states"],batch["states"]),0)
        valueBatch["targets"] =  torch.cat((previousBatch["returns"],batch["returns"]),0)

    def update_policy(self, batch):
        """ 
            Get advantage , states and action and calls trpo step
            Parameters:
            batch (dict of arrays of numpy)
            Returns:
        """
        advantages = batch["advantages"]
        states = batch["states"]
        actions = batch["actions"]
        trpo_step(self.policy_net, states,actions,advantages , self.max_kl, self.damping)

    def update_value(self, valueBatch):
        """ 
            Get valueBatch and run adam optimizer to learn value function
            Parameters:
            valueBatch  (dict of arrays of numpy)
            Returns:
        """
        # shuffle the data
        dataSize = valueBatch["targets"].size()[0]
        permutation = torch.randperm(dataSize)
        input = valueBatch["states"][permutation]
        target = valueBatch["targets"][permutation]

        iter = self.val_opt_iter
        batchSize = int(dataSize/ iter)

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        for t in range(iter):
            prediction = self.value_net(input[t*batchSize:t*batchSize+batchSize])
            loss = loss_fn(prediction, target[t*batchSize:t*batchSize+batchSize])
            # XXX : Comment out for debug
            # if t%100==0:
            #     print("\t%f"%loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def save_to_previousBatch(self, previousBatch, batch):
        """ 
            Save previous batch to use in future value optimization
            Parameters:
            previousBatch  (dict of arrays of numpy)
            batch (dict of arrays of numpy)
        """
        if self.value_memory<0:
            print("Value memory should be equal or greater than zero")
        elif self.value_memory>0:
            if previousBatch["returns"].size() == 0:
                previousBatch= {"states":batch["states"],
                                "returns":batch["returns"]}
            else:
                previous_size = previousBatch["returns"].size()[0]
                size =  batch["returns"].size()[0]
                if previous_size/size == self.value_memory:
                    previousBatch["states"] = torch.cat([previousBatch["states"][size:],batch["states"]],0)
                    previousBatch["returns"] = torch.cat([previousBatch["returns"][size:],batch["returns"]],0)
                else:
                    previousBatch["states"] = torch.cat([previousBatch["states"],batch["states"]],0)
                    previousBatch["returns"] = torch.cat([previousBatch["returns"],batch["returns"]],0)
        if self.value_memory_shuffle:
            permutation = torch.randperm(previousBatch["returns"].size()[0])
            previousBatch["states"] = previousBatch["states"][permutation]
            previousBatch["returns"] = previousBatch["returns"][permutation]

    def calculate_loss(self, reward_sum_mean,reward_sum_std,test_number = 10):
        """ 
            Calculate mean cummulative reward for test_nubmer of trials

            Parameters:
            reward_sum_mean (list): holds the history of the means.
            reward_sum_std (list): holds the history of the std.

            Returns:
            list: new value appended means
            list: new value appended stds
        """
        rewardSum = []
        for i in range(test_number):
            state = self.env.reset()
            rewardSum.append(0)
            for t in range(self.episode_length):
                state, reward, done, _ = self.env.step(self.policy_net.get_action(state)[0] )
                state = np.transpose(state)
                rewardSum[-1] += reward
                if done:
                    break
        reward_sum_mean.append(np.array(rewardSum).mean())
        reward_sum_std.append(np.array(rewardSum).std())
        return reward_sum_mean, reward_sum_std
    
    
    def predict(self, state):
        """
			Queries an action from the actor network, should be called from step.

			Parameters:
				state - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
		"""
        action = self.policy_net.get_action(state)[0]
        return action

                
    def learn(self, timesteps, print_every = 1):
        """
        Trains the agent

        Params
        ======
            timesteps (int): Number of timesteps the agent should interact with the environment
            print_every (int): Verbosity control
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        time_start = time.time()

        reward_sum_mean,reward_sum_std  = [], []
        previousBatch= {"states":torch.Tensor(0).to(self.device) ,
                        "returns":torch.Tensor(0).to(self.device)}

        reward_sum_mean,reward_sum_std = self.calculate_loss(reward_sum_mean,reward_sum_std)
        #print("Initial loss \n\tloss | mean : %6.4f / std : %6.4f"%(reward_sum_mean[-1],reward_sum_std[-1])  )
        num_steps = 0
        flag = False
        count_of_dones = 0
        
        max_iteration_number = timesteps//self.env.episode_length + 1
        
        for i_episode in range(max_iteration_number):
            time_episode_start = time.time()
            # reset batches
            batch = {"states":[] ,
                    "actions":[],
                    "next_states":[] ,
                    "rewards":[],
                    "returns":[],
                    "values":[],
                    "advantages":[],
                    "mask":[]}
            valueBatch = {"states" :[],
                        "targets" : []}


            # while num_steps < self.batch_size:
            done = False
            
            while not done:
                state = self.env.reset()
                reward_sum = 0
                states,actions,rewards,next_states,masks = [],[],[],[],[]
                steps = 0
                for t in range(self.env.episode_length):
                    action = self.policy_net.get_action(state)[0] # agent
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.transpose(next_state)
                    mask = 0 if done else 1

                    masks.append(mask)
                    states.append(state)
                    actions.append(action)
                    next_states.append(next_state)
                    rewards.append(reward)

                    state = next_state
                    reward_sum += reward
                    num_steps+=1
                    
                if done:
                    count_of_dones += 1
                    flag = True
            
                if flag and count_of_dones % print_every == 0:
                    print(f'Score at timestep {num_steps}: {reward_sum}.')
                    flag = False
                    
                if num_steps >= timesteps:
                    break

            batch["states"].append(np.expand_dims(states, axis=1) )
            batch["actions"].append(actions)
            batch["next_states"].append(np.expand_dims(next_states, axis=1))
            batch["rewards"].append(rewards)
            batch["mask"].append(masks)
            #num_steps += steps

            self.prepare_data(batch,valueBatch,previousBatch)
            self.update_policy(batch) # First policy update to avoid overfitting
            self.update_value(valueBatch)

            self.save_to_previousBatch(previousBatch,batch)

            #print("episode %d | total: %.4f "%( i_episode, time.time()-time_episode_start))
            reward_sum_mean,reward_sum_std = self.calculate_loss(reward_sum_mean,reward_sum_std)
            #print("loss | mean : %6.4f / std : %6.4f"%(reward_sum_mean[-1],reward_sum_std[-1]))
