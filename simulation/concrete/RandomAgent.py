from simulation.base.Agent import Agent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.base.State import State
class RandomAgent(Agent):
    """
    Random agent, returns a single action for 200 steps and then goes on to another random action
    """
    def __init__(self):
        self.curr_action = SimulDiscreteAction.sample()
        # self.curr_action = SimulDiscreteAction.LEFT
        self.counter = 0
    def get_action(self, s: State) -> SimulDiscreteAction:
        # print(s.get_value())
        """
        This should return a single action from the information in s.
        """
        self.counter += 1
        if self.counter > 200:
            self.counter = 0
            self.curr_action = SimulDiscreteAction.sample()
        return self.curr_action
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float):
        # don't do anything
        if reward == -2: # in wall, turn in other dir
            self.curr_action = SimulDiscreteAction.sample()
            self.counter = 0
        pass