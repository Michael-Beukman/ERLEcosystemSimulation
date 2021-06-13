from simulation.main.Entity import EntityType
# from simulation.base.Agent import Agent
from simulation.multi_agent_sim.MultiAgent import MultiAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.base.State import State
import numpy as np
# order of actions based on new order of sectors in discrete
mapping = [7, 0, 1, 6, 2, 5, 4 ,3]
class HardCodePred(MultiAgent):
    def get_action(self, s: State) -> SimulDiscreteAction:
        v = s.get_value()
        best = (-100000000, -1)
        # print(v)
        scores = []
        for i in range(8):
            other = (i + 4) % 8
            # [food, prey, pred, wall]
            score = 1 * v[2 + i*4  + 1] + -100 * v[2 + i * 4 + 3] + 50 * v[2 + other * 4 + 3]
            scores.append(score)
            best = max(best, (score, i))
        scores = np.array(scores)
        id = scores == scores.max()
        v = np.arange(len(scores))
        indices = v[id]
        index_to_choose = np.random.choice(indices)
        thingy = mapping[index_to_choose]
        act = SimulDiscreteAction(thingy)
        return act

    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float, next_state: State):
        return super().add_sample(s, a, reward, next_state=next_state)


class HardCodePrey(MultiAgent):
    def get_action(self, s: State) -> SimulDiscreteAction:
        v = s.get_value()
        best = (-100000000, -1)
        # print(v)
        scores = []
        for i in range(8):
            other = (i + 4) % 8
            # [food, prey, pred, wall]
            score = 1 * v[2 + i*4] + -100 * v[2 + i * 4 + 3] + 50 * v[2 + other * 4 + 3] \
                    -1000 * v[2 + i * 4 + 2] + 500 * v[2 + other * 4 + 2]
            scores.append(score)
            best = max(best, (score, i))
        scores = np.array(scores)
        id = scores == scores.max()
        v = np.arange(len(scores))
        indices = v[id]
        index_to_choose = np.random.choice(indices)
        # print(s.get_value(), scores, index_to_choose, mapping[index_to_choose], SimulDiscreteAction(mapping[index_to_choose])) 
        thingy = mapping[index_to_choose]
        act = SimulDiscreteAction(thingy)
        return act
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float, next_state: State):
        
        return super().add_sample(s, a, reward, next_state=next_state)

class RandomMultiAgent(MultiAgent):
    def __init__(self) -> None:
        pass
    def get_action(self, s: State) -> SimulDiscreteAction:
        return SimulDiscreteAction.sample()
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float, next_state: State):
        pass