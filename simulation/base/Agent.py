from abc import ABC, abstractmethod
from simulation.base.Action import Action
from simulation.base.State import State
class Agent(ABC): 
    """
    This is the base agent class to extend. It can get an action & add a sample.
    The learning gets implemented in the add_sample function.
    """
    main_id = 0
    def __init__(self) -> None:
        self.id = Agent.main_id
        Agent.main_id +=1
        super().__init__()

    @abstractmethod
    def get_action(self, s: State) -> Action:
        """
        This should return a single action from the information in s.
        """
        raise NotImplemented
    
    @abstractmethod
    def add_sample(self, s: State, a: Action, reward: float, next_state: State=None):
        pass
if __name__ == '__main__':
    # print(help(Agent))
    a1 = Agent()
    a2 = Agent()
    a3 = Agent()
    print(a1.id, a2.id, a3.id)
    # action = get-action()
    # next_state = step(action)
    # add_sample(..)
    # class SBAgent:
    #     def get_action():
    #         return max(weights * state)
        
    #     def add_sample():


    #     def learn():
    #         for i in episode:
    #             action = self.get_action()
    #             next_state = step(action)
    #             self.add_sample(..)
    pass