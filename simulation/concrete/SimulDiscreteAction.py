from enum import Enum, auto, EnumMeta
from abc import ABCMeta
from simulation.base.Action import Action
import numpy as np
class SimulDiscreteAction(Action, Enum):
    """
    A specific action that an agent can take. It can move in one of the 8 cardinal directions.
    """
    UP = 0
    UP_RIGHT = 1
    RIGHT = 2
    DOWN_RIGHT = 3
    DOWN = 4
    DOWN_LEFT = 5
    LEFT = 6
    UP_LEFT = 7



    @classmethod
    def sample(cls: "SimulDiscreteAction") -> "Action":
        return np.random.choice(list(cls))

if __name__ == '__main__':
    action = SimulDiscreteAction.LEFT
    print(action, SimulDiscreteAction.sample(), SimulDiscreteAction(1))
    print(action.value)
    pass