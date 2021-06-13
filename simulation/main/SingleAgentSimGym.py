import gym
from agent.rl.simple_q_learning import SimpleQLearningAgent
from simulation.base.Action import Action
from typing import List, Tuple
from simulation.main.Entity import Entity, EntityType
from simulation.base.Agent import Agent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.concrete.RandomAgent import RandomAgent
from simulation.concrete.HardCodeAgentDistanceSectors import HardCodeAgentDistanceSectors
import simulation.concrete.StateRepresentations as StateRep
import numpy as np
from simulation.main.Simulation import SingleAgentSimulation

class SingleAgentSimGym(gym.Env):
    def __init__(self):
        self.simulation = self.get_sim()

        self.action_space = gym.spaces.Discrete(len(SimulDiscreteAction))
        low = np.zeros(self.simulation.state_rep.get_num_features())
        # for velocities.
        low[0] = -1
        low[1] = -1
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(low, high)
        self.dist_indices = None
    
    def seed(self, seed):
        np.random.seed(seed)

    def get_sim(self):
        return SingleAgentSimulation.basic_simul(StateRep.StateRepSectorsWithDistAndVelocity())
        
    def step(self, action: int):
            reward = self.simulation.update(SimulDiscreteAction(action), self.dist_indices)
            done = False
            info = {}
            state, self.dist_indices = self.simulation._get_state(0)

            return state.get_value(), reward, done, info
            
    def reset(self):
        self.simulation = self.get_sim()
        state, self.dist_indices = self.simulation._get_state(0)
        return state.get_value()
