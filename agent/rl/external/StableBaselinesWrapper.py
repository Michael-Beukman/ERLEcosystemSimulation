from simulation.base.Agent import  Agent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
# from torch import nn
# import torch
import numpy as np
from stable_baselines3 import PPO, DQN

class StableBaselines(Agent):
    def __init__(self, name='./models/ppo_rl', is_ppo=None):
        if is_ppo is None:
            if 'dqn' in name.lower():
                self.model = DQN.load(name)
            else:
                self.model = PPO.load(name)
        else:
            self.model = PPO.load(name) if is_ppo else DQN.load(name)
        pass
    
    def get_action(self, s: State) -> SimulDiscreteAction:
        ans, _ = self.model.predict(s.get_value(), deterministic=False)
        return SimulDiscreteAction(ans)

    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float):
        pass