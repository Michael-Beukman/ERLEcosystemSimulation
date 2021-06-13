import gym
gym.envs.register(
    id='SingleAgentSim-v0',
    entry_point='simulation.main.SingleAgentSimGym:SingleAgentSimGym',
    max_episode_steps=4000,
)
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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'models')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) == 0:
              mean_reward = 0
          else:  
            mean_reward = np.mean(y[-100:])
          mean_reward = int(mean_reward)
          print(f"Mean reward = {mean_reward} at ep {self.n_calls}")
          self.model.save(self.save_path + "/" + str(self.n_calls) + "_" +str(mean_reward))
        return True

def eval(model, name):
    env = gym.make("SingleAgentSim-v0")
    total_r  = 0
    for ep in range(10):
        print(ep)
        r = 0
        done = False
        state = env.reset()
        while not done:
            a, _ = model.predict(state)
            state, r, done, _ = env.step(a)
            total_r += r
    print(f"Total average reward over 100 episodes {name}", total_r/10)

def do_all(name='ppo'):
    log_dir = f"./pickles/experiment1/{name}/proper_v1"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('SingleAgentSim-v0')
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)
    if name == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = DQN("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=1e7, callback=SaveOnBestTrainingRewardCallback(1e5, log_dir=log_dir))

    model.save(f"{log_dir}/models/final_1e7")
    eval(model, name)