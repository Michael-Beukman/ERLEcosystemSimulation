from simulation.main.Entity import EntityType
from simulation.base.Agent import Agent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.base.State import State
import numpy as np
import copy

class MultiAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.prev_state = None

    def get_action(self, s: State) -> SimulDiscreteAction:
        return super().get_action(s)
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float, next_state: State):
        self.prev_state = next_state
        return super().add_sample(s, a, reward, next_state=next_state)
    
    def reproduce(self) -> "MultiAgent":
        return copy.deepcopy(self)