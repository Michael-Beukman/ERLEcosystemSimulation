import gym
from stable_baselines3.common.vec_env import DummyVecEnv
import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
gym.envs.register(
    id='SingleAgentSim-v0',
    entry_point='simulation.main.SingleAgentSimGym:SingleAgentSimGym',
    max_episode_steps=4000,
)
env = gym.make('SingleAgentSim-v0')
model_ppo = PPO("MlpPolicy", env, verbose=1)
model_dqn = DQN("MlpPolicy", env, verbose=1)

model_ppo.save(f"./pickles/experiment1/ppo/proper_v1/at_0_steps")
model_dqn.save(f"./pickles/experiment1/dqn/proper_v1/at_0_steps")
