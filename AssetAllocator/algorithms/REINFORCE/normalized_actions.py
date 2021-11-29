import gym
from gym import spaces
import numpy as np


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return actions

# https://github.com/google-research/google-research/blob/master/algae_dice/wrappers/normalize_action_wrapper.py
class NormalizeBoxActionWrapper(gym.ActionWrapper):
  """Rescale the action space of the environment."""

  def __init__(self, env):
    if not isinstance(env.action_space, spaces.Box):
      raise ValueError('env %s does not use spaces.Box.' % str(env))
    super(NormalizeBoxActionWrapper, self).__init__(env)
    self._max_episode_steps = env.max_episode_steps

  def action(self, action):
    # rescale the action
    low, high = self.env.action_space.low, self.env.action_space.high
    scaled_action = low + (action + 1.0) * (high - low) / 2.0
    scaled_action = np.clip(scaled_action, low, high)

    return scaled_action

  def reverse_action(self, scaled_action):
    low, high = self.env.action_space.low, self.env.action_space.high
    action = (scaled_action - low) * 2.0 / (high - low) - 1.0
    return action


def check_and_normalize_box_actions(env):
  """Wrap env to normalize actions if [low, high] != [-1, 1]."""
  low, high = env.action_space.low, env.action_space.high

  if isinstance(env.action_space, spaces.Box):
    if (np.abs(low + np.ones_like(low)).max() > 1e-6 or
        np.abs(high - np.ones_like(high)).max() > 1e-6):
        print('Normalizing environment actions.')
        return NormalizeBoxActionWrapper(env)

  # Environment does not need to be normalized.
  return env