import os
import torch
from argparse import ArgumentParser

from AssetAllocator.experiment import Experiment

model_names = [
    'TD3', 'NAF', 'PPO', 'TRPO', 'DDPG', 'REINFORCE', 'SAC', 'A2C',
    'STB-TD3', 'STB-SAC', 'STB-PPO', 'STB-A2C', 'STB-DDPG',
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # keyword arguments to pass to trainer class (remember to change test_runs for stochastic models)
    trainer_kw = {'print_every': 1, 'test_runs': 1}

    # keyword arguments to pass to your models (during hyperparameter tuning)

    model_kw = {'device': device}

    exp = Experiment(trainer_kwargs=trainer_kw, model_kwargs=model_kw)

    exp.run('SAC')


if __name__ == "__main__":
    main()
