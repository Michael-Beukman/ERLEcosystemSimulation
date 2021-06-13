from simulation.base.Agent import Agent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.base.State import State
class RandomAgentDumb(Agent):
    """
    Random agent, chooses different random action at each step.
    """
    def __init__(self):
        pass
    
    def get_action(self, s: State) -> SimulDiscreteAction:
        # print(s.get_value())
        """
        This should return a single action from the information in s.
        """
        return SimulDiscreteAction.sample()
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float):
        pass