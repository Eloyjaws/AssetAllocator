import os
import pickle
import sys

if '__file__' in vars():
    # print("We are running the script non interactively")
    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)    
else:
    # print('We are running the script interactively')
    sys.path.append("..")

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .environments.PortfolioGym import PortfolioManagementGym as PMG
from .environments.utils import *

def load_stock_data(filepath):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(dir_path + '/' + filepath, index_col = 'Date')
    for col in data.columns:
       data[col] = data[col].astype(float)
    
    idx = int(len(data) * 0.7)
    train_data = data.iloc[:idx]
    test_data = data.iloc[idx:]
    #print(len(train_data), len(test_data))
    return train_data, test_data

def evaluate(test_env, model, test_length):
    agent_rwds = []
    obs = test_env.reset()
    dones = False

    while not dones:
        action = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        agent_rwds  += [test_env.render()]

    #simple_returns = [agent_rwd[0] for agent_rwd in agent_rwds]
    rwds = agent_rwds[-test_length:]
    #print(len(rwds))
    return rwds

class Trainer:
    def __init__(self, 
                 filepath, 
                 experiment_name, 
                 timesteps, 
                 print_every,
                 episode_length = None,
                 returns = True,
                 trading_cost_ratio = 0.001,
                 lookback_period = 64,
                 initial_investment = 1_000_000,
                 retain_cash = True,
                 random_start_range = 20,
                 dsr_constant = 1e-4,
                 add_softmax = False,
                 start_date = '2009-01-01',
                 end_date = '2022-01-01',
                 seed = 0,
                 test_length = 550,
                 test_runs = 1):
        
        self.filepath = filepath
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.print_every = print_every
        self.episode_length = episode_length 
        self.returns = returns
        self.trading_cost_ratio = trading_cost_ratio
        self.lookback_period = lookback_period
        self.initial_investment = initial_investment
        self.retain_cash = retain_cash
        self.random_start_range = random_start_range
        self.dsr_constant = dsr_constant
        self.add_softmax = add_softmax
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        self.test_length = test_length
        self.test_runs = test_runs
        
    def get_train_env(self):
        self.train_data, self.test_data = load_stock_data(self.filepath)

        self.train_env = PMG(
                             data = self.train_data,
                             episode_length = len(self.train_data),
                             returns = self.returns,
                             trading_cost_ratio = self.trading_cost_ratio,
                             lookback_period = self.lookback_period,
                             initial_investment = self.initial_investment,
                             retain_cash = self.retain_cash,
                             random_start_range = self.random_start_range,
                             dsr_constant = self.dsr_constant,
                             add_softmax = self.add_softmax,
                             start_date = self.start_date,
                             end_date = self.end_date,
                             seed = self.seed)
        return self.train_env
        
    def run(self, model, train_mode = True):
        results = []
        
        if train_mode:
            try:
                model.learn(timesteps = self.timesteps, print_every = self.print_every)
            except TypeError:
                model.learn(total_timesteps = self.timesteps, log_interval = self.print_every)


        # Run the test phase for different phase
        for seed in tqdm(range(self.test_runs)):

            test_env =  PMG(
                             data = self.test_data,
                             episode_length = len(self.test_data),
                             returns = self.returns,
                             trading_cost_ratio = self.trading_cost_ratio,
                             lookback_period = self.lookback_period,
                             initial_investment = self.initial_investment,
                             retain_cash = self.retain_cash,
                             random_start_range = 0,
                             dsr_constant = self.dsr_constant,
                             add_softmax = self.add_softmax,
                             start_date = self.start_date,
                             end_date = self.end_date,
                             seed = self.seed
                            )
            
            results += [evaluate(test_env, model, self.test_length)]

        # Save the results for each analysis into different files
        with open(f'{self.experiment_name}.pkl', 'wb') as f:
            pickle.dump(results, f)

        with open(f'{self.experiment_name}_portfolio.pkl', 'wb') as f:
            pickle.dump(test_env.weights, f)
    
        return results
        
        
        