import gym
gym.envs.register(
    id='SingleAgentSim-v0',
    entry_point='simulation.main.SingleAgentSimGym:SingleAgentSimGym',
    max_episode_steps=4000,
)

import gym
import json
import datetime as dt
from stable_baselines3.common.vec_env import DummyVecEnv
import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import stable_baselines3.common.results_plotter as results_plotter
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
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True
# Create log dir
log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('SingleAgentSim-v0')
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)
print("Hey there")
# The algorithms require a vectorized environment to run
model = DQN("MlpPolicy", env, verbose=1)
# model = PPO.load("models/ppo_rl", env)
for i in range(1000):
  print("STEP ", i)
  model.learn(total_timesteps=1e4)
  if i == 0 or i == 20 or (i+1) % 200 == 0:
    model.save(f'./models/0421/DQN_0_vel_rl_{(i+1) * 1e4}')

# model.save(f'./models/0420/DQN_MASSIVE_10mil')
model.save(f'./models/0421/_DQN_0_vel_rl_{(i+1) * 1e4}')

# Helper from the library
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO")
# plt.savefig('./plots/PPO_0_vel_1mil_again_with_velocitylots.png')
# model.save("./models/PPO_0_vel_rl_large_with_velocity_1mil")

def eval(model):
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
    print("Total average reward over 100 episodes DQN", total_r/100)

eval(model)