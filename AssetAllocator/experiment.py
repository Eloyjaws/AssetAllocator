import os
import sys

if '__file__' in vars():
    # print("We are running the script non interactively")
    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)    
else:
    # print('We are running the script interactively')
    sys.path.append("..")

from .algorithms.Baselines.agent import BaselineAgent
from .algorithms.NAF.agent import NAFAgent
from .algorithms.TD3.agent import TD3Agent

from .algorithms.PPO.agent import PPOAgent
from .algorithms.TRPO.agent import TRPOAgent

from .algorithms.DDPG.agent import DDPGAgent
from .algorithms.REINFORCE.agent import REINFORCEAgent

from .algorithms.SAC.agent import SACAgent
from .algorithms.A2C.agent import A2CAgent

from .environments.utils import log_to_simple, simple_to_log
from .trainer import Trainer

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import TD3 as STBTD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import SAC as STBSAC
from stable_baselines3 import PPO as STBPPO
from stable_baselines3 import DDPG as STBDDPG
from stable_baselines3 import A2C as STBA2C

class Experiment:
    def __init__(self, trainer_kwargs = {}, model_kwargs = {}, timesteps = None):
        self.trainer_kwargs = trainer_kwargs
        self.model_kwargs = model_kwargs
        self.timesteps = timesteps
        
    def run(self, model_name, dataset = None):
        if dataset is None:
            dataset = 'DOW30'
        
        if self.timesteps is None:
            timesteps = [10_000, 100_000]
        else:
            timesteps = self.timesteps
        
        rewards = [True] #[True, False]
        trading_costs = [0, 0.001, 0.01]
        retain_cash = False
        lookback = 64
        
        if model_name in ['MPT', 'Uniform', 'Random', 'BuyAndHold']:
            for trading_cost in trading_costs:
                path = f'./Baseline_{dataset}_Results/'  
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path) 
                name = f'{path}{model_name}-Trading_cost-{trading_cost}'
                
                if model_name == 'Random':
                    self.trainer_kwargs['test_runs'] = 100

                trainer = Trainer(filepath = f'data/{dataset}.csv',
                              experiment_name = name,
                              timesteps = -1,
                              lookback_period = lookback,
                              trading_cost_ratio = trading_cost,
                              **self.trainer_kwargs)

                env = trainer.get_train_env()
                model = BaselineAgent(model_name, env, **self.model_kwargs)
                    
                returns = trainer.run(model)

                log_rets = 0

                for ret in returns:
                        log_rets += log_to_simple(sum([simple_to_log(i[0]) for i in ret]))

                log_rets /= len(returns)
                n = name.split('/')[-1]
                print(f'{n} : {log_rets}')

        else:
            for reward in rewards:
                for timestep in timesteps:
                    for trading_cost in trading_costs:                                            
                        if reward:
                            strng = 'LogRets'
                        else:
                            strng = 'ShRt'

                        path = f'./{model_name}_{dataset}_Results/'
                        isExist = os.path.exists(path)
                        if not isExist:
                            os.makedirs(path)                       

                        name = f'{path}Reward-{strng}_timestep-{timestep}_trading_cost-{trading_cost}'

                        trainer = Trainer(filepath = f'data/{dataset}.csv',
                              experiment_name = name,
                              timesteps = timestep,
                              lookback_period = lookback,
                              trading_cost_ratio = trading_cost,
                              returns = reward,
                              retain_cash = retain_cash,
                              **self.trainer_kwargs)

                        env = trainer.get_train_env()
                                         
                        if model_name in ['TD3']:
                            model = TD3Agent(env, **self.model_kwargs)
                        elif model_name in ['NAF']:
                            model = NAFAgent(env, **self.model_kwargs)
                        elif model_name in ['PPO']:
                            model = PPO(env, **self.model_kwargs)
                        elif model_name in ['TRPO']:
                            #env = SubprocVecEnv([env  for i in range(4)])
                            model = TRPOAgent(env, **self.model_kwargs)
                        elif model_name in ['DDPG']:
                            model = DDPGAgent(env, **self.model_kwargs)
                        elif model_name in ['REINFORCE']:
                            model = REINFORCEAgent(env, **self.model_kwargs)
                        elif model_name in ['SAC']:
                            model = SACAgent(env, **self.model_kwargs)
                        elif model_name in ['A2C']:
                            model = A2CAgent(env, **self.model_kwargs)
                        elif model_name in ['STB-TD3']:
                            n_actions = env.action_space.shape[-1]
                            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                            model = STBTD3("MlpPolicy", env, action_noise=action_noise, verbose=1, **self.model_kwargs)
                        elif model_name in ['STB-SAC']:
                            model = STBSAC("MlpPolicy", env, verbose=1, **self.model_kwargs)
                        elif model_name in ['STB-PPO']:
                            #env = SubprocVecEnv([env  for i in range(4)])
                            model = STBPPO("MlpPolicy", env, verbose=1)
                        elif model_name in ['STB-A2C']:
                            #env = SubprocVecEnv([env  for i in range(4)])
                            model = STBA2C("MlpPolicy", env, verbose=1)
                        elif model_name in ['STB-DDPG']:
                            n_actions = env.action_space.shape[-1]
                            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                            model = STBDDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, **self.model_kwargs)
                        else:
                            assert False, 'Wrong Name Passed In!'

                        returns = trainer.run(model)

                        log_rets = 0

                        for ret in returns:
                                log_rets += log_to_simple(sum([simple_to_log(i[0]) for i in ret]))

                        log_rets /= len(returns)
                        n = name.split('/')[-1]
                        print(f'{n} : {log_rets}')