import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import pandas as pd
from .utils import softmax, log_to_simple

import sys
sys.path.append('../')

import yfinance as yf

class PortfolioManagementGym(gym.Env):
    """
    Portfolio Management Gym
    """
    def __init__(self,
                 data,
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
                 seed = 0):
        """
        Initializes the gym environment

        Args:
            data : pandas dataframe with date index and stock columnns with price data 
                or list of stock tickers
                
            episode_length : how long the agent should interact with the environment

            returns: If True, use log_returns as reward. Else, use sharpe ratio

            trading_cost_ratio : percentage of stock price that accounts for trading costs

            lookback_period : a fixed sized window, used to know how much data to return to the agent as observation

            initial_investment : how much the agent wants to invest

            retain_cash : bool value to tell the value whether to keep a cash value.

            random_start_range : random start position for training, should be set to 0 for test

            dsr_constant : smoothing parameter for differential sharpe ratio

            add_softmax : bool value to tell the agent whether to soft-normalize the input action

            start_date : start date for yahoo finance download

            end_date : end date for yahoo finance download

            seed : seed value for environment reproducibility
                  
        """

        self.data = data
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

        if isinstance(data, pd.DataFrame):
            self.stocks_names = list(self.data.columns)
            _, self.n = self.data.shape
        else:
            self.stocks = data
            self.n = len(self.data)
        
        action_dim = self.n + self.retain_cash
        state_dim = self.lookback_period * self.n
        
        self.start_date = start_date
        self.end_date = end_date
        
        self.observation_space = spaces.Box(np.finfo('d').min,np.finfo('d').max,shape=(state_dim,))
        self.action_space = spaces.Box(0, 1, shape=(action_dim,))
        
        self._seed(seed)
        self.reset()

    def reset(self):
        """
        Resets the environment to the start state

        Returns:
            Initial observation (array_like)
        """        
        self._initialize_env()
        obs = self._get_new_state()
        width = self.observation_space.shape[0] - len(obs)
        obs = np.array(np.pad(obs, (width,0), constant_values = 0), dtype = 'float32')
        return obs

    def step(self, action):
        """
        Takes in an action of size action_dim
        Returns the observation, reward, episode_statuss
        """

        if not self._get_done_status():
            self.num_actions_taken += 1
            self._take_action(action)
        
        new_state = np.array(self._get_new_state(), dtype = 'float32')
        assert self.observation_space.contains(new_state), \
            f'observation does not belong to space'

        reward = self._get_reward()
        episode_over = self._get_done_status()
        info = {}

        return new_state, reward, episode_over, info

    def render(self):
        """
        Returns an array of simple returns, differential sharpe ratio, and available amount
        """        
        return [log_to_simple(self.log_returns[-1]), 
                    self.sharpe_ratios[-1], self.AVAILABLE_AMT]


   #################################################################################################
    #################################### HELPER FUNCTIONS ###########################################
    #################################################################################################
    

    def _seed(self, seed = None):
        """
        Helper method to set seed
        """        
        self.np_random, self.seed = seeding.np_random(seed)


    def _load_data(self):
        """
        Helper method to load the data
        """
        assert isinstance(self.data, pd.DataFrame) or isinstance(self.data, list), \
        'Please provide a list of tickers or a dataframe'

        # downloading data from yahoo finance if tickers were provided
        if isinstance(self.data, list):
            prices = yf.download(self.data, start = self.start_date,
                                    end = self.end_date, interval="1d", actions=True)
            prices.dropna(inplace=True)
            prices = prices["Adj Close"]
        else:
            prices = self.data.copy()
            prices.dropna(inplace=True)
        
        prices.index = pd.to_datetime(prices.index)
        
        for col in prices.columns:
            prices[col] = prices[col].astype(np.float32)
        return prices
    
    def _preprocess_data(self, df):
        """
        Helper method to preprocess the data
        """
        date_range = range(len(df))
        index_to_date_map = dict(zip(date_range, df.index))
        returns_df = df.pct_change().fillna(0)
        return index_to_date_map, returns_df

    def _initialize_env(self):
        """
        Helper method to create all the environment variables
        """
        ext_stock_list = self.stocks_names.copy() + ['Cash', 'Trading Costs']
        self.weights = [dict(zip(ext_stock_list, [0]*(self.n) + [1, 0]))]
        self.current_holding = [0] * self.n
        self.AVAILABLE_AMT = self.initial_investment
        self.CASH = self.initial_investment

        self.num_actions_taken = 0
        self.curr_reward = 0
        self.log_returns = []
        self.sharpe_ratios = []
        self.date_map = None

        self.prices = self._load_data()
        self.date_map, self.observations = self._preprocess_data(self.prices)
        
        self.start_day = self.np_random.choice(range(self.lookback_period, 
                            self.lookback_period + self.random_start_range + 1))
        self.end_day = min(self.start_day + self.episode_length, 
                            len(self.prices) - 1)

    def _get_new_state(self):
        """
        Helper method to return current observation state
        """
        self.current_day = self.num_actions_taken + self.start_day
        state_end_day = self.current_day - 1
        state_start_day = state_end_day - self.lookback_period + 1

        state_start_date = self.date_map[state_start_day]
        state_end_date = self.date_map[state_end_day]


        state_obs = self.observations[state_start_date : state_end_date]
        state_obs = np.array(np.concatenate(state_obs.fillna(0).values, 
                    axis = 0), dtype = 'float32')
        return state_obs

    def _check_actions(self, action, check_dim = True):
        """
        Helper method to check validity of the actions received
        """
        if abs(sum(action) - 1) > 1e-3:
            print(action)
            assert False, 'Wrong portfolio weights!'
        
        if check_dim:
            assert self.action_space.contains(np.array(action, dtype = 'float32')), \
                    f'{action} action does not belong to space'

    
    def _compute_buyable_shares(self, budgets, prices):
        """Helper method to compute buyable shares
        """        
        shares = [budget/price for budget, price in zip(budgets, prices)]
        return shares
    
    def _compute_trading_costs(self, shares_now, shares_prev, prices):
        """
        Helper method to compute trading costs
        """        
        trading_costs = []
        
        for now, prev, price in zip(shares_now, shares_prev, prices):
            diff = abs(now - prev)
            if diff < 1:
                trading_costs.append(0)
            else:
                #print(diff, price, self.trading_cost_ratio)
                trading_costs.append(diff * price * self.trading_cost_ratio)
        
        return trading_costs

    def _take_action(self, actions):
        """
        Helper method to compute effects of agent's action on environment
        """
        # For stable baseline model implementations
        if isinstance(actions, tuple):
            actions = actions[0]

        if self.add_softmax:
            actions = softmax(actions)

        self._check_actions(actions)
        
        # Allocating Budget
        if self.retain_cash:
            actions_ = actions[:-1]
            cash_budget_ratio = actions[-1]
        else:
            actions_ = actions
            cash_budget_ratio = 0

        budget_allocation = [action * self.AVAILABLE_AMT \
                            for action in actions_]
        self.CASH = cash_budget_ratio * self.AVAILABLE_AMT

        # Computing Trading Costs
        current_date = self.date_map[self.current_day]
        prices = self.prices.loc[current_date]
        
        buyable_shares = self._compute_buyable_shares(
                                                      budget_allocation,
                                                      prices)

        trading_costs = self._compute_trading_costs(buyable_shares, 
                                                    self.current_holding,
                                                    list(prices.values))

        self.current_holding = buyable_shares

        # Recomputing portfolio weights
        total_trading_costs = sum(trading_costs)
        ext_stock_list = self.stocks_names.copy() + ['Cash', 'Trading Costs']
        ext_allocation = budget_allocation + [self.CASH, total_trading_costs]
        total_amount = sum(ext_allocation)
        
        portfolio_weights = {stock : allocation/total_amount \
                            for stock, allocation in \
                            zip(ext_stock_list, ext_allocation)}

        self._check_actions(list(portfolio_weights.values()), check_dim = False)
        self.weights.append(portfolio_weights)
        
        assert total_trading_costs >= 0, 'Error in trading costs calculations'
        self.AVAILABLE_AMT = total_amount - total_trading_costs
        

    def _get_returns(self):
        """
        Helper method to calculate log returns
        """
        current_date = self.date_map[self.current_day]
        observation = self.observations.loc[current_date]
        curr_portfolio = self.weights[self.num_actions_taken]
        
        
        simple_returns = 0
        for stock in self.stocks_names:
            simple_returns += curr_portfolio[stock] * observation[stock]
        
        a =  simple_returns * self.AVAILABLE_AMT
        self.AVAILABLE_AMT += a
        
        self.log_returns += [np.log(simple_returns + 1)]
        
    def _get_sharpe_ratio(self):
        """
        Helper method to calculate differential sharpe ratio
        """
        if self.num_actions_taken < self.lookback_period:
            S = 0
        else:
            window = [log_to_simple(i) for i in self.log_returns]
            S = np.nanmean(window)/np.nanstd(window) * np.sqrt(252) / len(window)
        self.sharpe_ratios += [S]

    def _get_reward(self):
        """
        Helper method to calculate both rewards and return one 
        """
        self._get_returns()
        self._get_sharpe_ratio()
        
        if self.returns:
            curr_reward = self.log_returns[-1]
        else:
            curr_reward = self.sharpe_ratios[-1]
        return curr_reward

    def _get_done_status(self):
        """
        Helper method to get end state status
        """
        return self.current_day >= self.end_day