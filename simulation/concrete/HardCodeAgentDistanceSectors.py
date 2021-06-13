from simulation.main.Entity import EntityType
from simulation.base.Agent import Agent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.base.State import State
import numpy as np

class HardCodeAgentDistanceSectors(Agent):
    """Hard coded single agent that uses the standard staterepresentation."""

    def get_action(self, s: State) -> SimulDiscreteAction:
        v = s.get_value()
        best = (-100000000, -1)
        # print(v)
        scores = []
        for i in range(8):
            other = (i + 4) % 8

            # go towards food, and away from walls.
            score = 1 * v[2 + i*2] + -100 * v[2 + i * 2 + 1] + 50 * v[2 + other * 2 + 1]
            # scores.append((score, i))
            scores.append(score)
            best = max(best, (score, i))
        scores = np.array(scores)
        id = scores == scores.max()
        # print(id, scores[id])
        v = np.arange(len(scores))
        indices = v[id]
        if len(scores) != 4 or 1:
            index_to_choose = np.random.choice(indices)
        else:
            index_to_choose = (indices)[0]

        act = SimulDiscreteAction(index_to_choose)
        return act
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float):
        # Nothing here
        return super().add_sample(s, a, reward)