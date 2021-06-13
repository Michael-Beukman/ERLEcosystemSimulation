from typing import Tuple

from stable_baselines3.dqn.policies import MlpPolicy
from agent.rl.external.StableBaselinesWrapper import StableBaselines
import pickle
import gym
from simulation.base.Agent import Agent
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
        self.all_training_rewards = []
        self.checks = 1e5
        self.allx = None
        self.ally = None


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(y):
            self.all_training_rewards.append(y[-1])
        if self.n_calls % self.checks == 0:
          if len(x) == 0:
              mean_reward = 0
          else:  
            mean_reward = np.mean(y[-100:])
          mean_reward = int(mean_reward)
          print(f"Mean reward = {mean_reward} at ep {self.n_calls}")
          self.model.save(self.save_path + "/" + str(self.n_calls) + "_" +str(mean_reward))
        self.allx = x; self.ally = y;
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

# def proper_train(name='ppo'):
def proper_train(name: str, train_model_dir, num_steps) -> Tuple[Agent, np.ndarray]:
    log_dir = train_model_dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('SingleAgentSim-v0')
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)
    if name == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = DQN("MlpPolicy", env, verbose=1)

    cb = SaveOnBestTrainingRewardCallback(min(num_steps//2, 1e5), log_dir=log_dir);
    model.learn(total_timesteps=num_steps, callback=cb)

    dir = f"{log_dir}/models/final_{num_steps}"
    model.save(dir)
    with open(os.path.join(log_dir, "final_save_rewards"), 'wb+') as f:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        pickle.dump({
            # 'all_rewards': cb.all_training_rewards,
            'all_xs': x,
            'all_ys': y,
        }, f)
    agent_to_return = StableBaselines(dir, name.upper() == "PPO")
    return agent_to_return, y

