from simulation.base.Agent import  Agent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from torch import nn
import torch
import numpy as np
class EvolutionaryAgent:
    """
        This is a single agent that takes part in the genetic algorithm
        Predicts actions by performing argmax(weight @ state) where state is of size S, the number of actions is A 
        and weight is of shape A x S.
    """
    def __init__(self, num_obs:int, num_actions:int):
        """Creats

        Args:
            num_obs (int): State dim.
            num_actions (int): Number of actions
        """
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.network =  nn.Sequential(
                                nn.Linear(num_obs, num_actions),
                                # We perform a softmax here but that doesn't matter too much since we always take the largest value.
                                nn.Softmax()
                                )
        self.fitness = -1
    def get_action(self, s: State) -> SimulDiscreteAction:
        with torch.no_grad():
            ss = torch.from_numpy(s.get_value().astype(np.float32))#.double()
            ans = self.network(ss)
        # Chooses best action (if multiple with same score, choose randomly)
        index_to_choose = torch.multinomial(ans, 1)
        best_index = index_to_choose

        return SimulDiscreteAction(best_index.numpy())
    
    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float):
        # No add sample.
        pass
    
    def crossover(self, other: "EvolutionaryAgent") -> "EvolutionaryAgent":
        """
        This performs a crossover operation with another agent and returns a child.
        It basically takes some weights from self and some from others.

        Returns:
            EvolutionaryAgent: The child to return
        """
        # Create new child
        child = EvolutionaryAgent(self.num_obs, self.num_actions)
        with torch.no_grad():
            # For all params,
            for name, param in self.network.named_parameters():
                # Get a list of True's and False's
                indices = torch.rand_like(param) > 0.5
                # temporary var.
                tmp = torch.zeros_like(param)
                # randomly either take parent 1 or parent 2's values
                tmp[indices] = param[indices]
                tmp[~indices] = dict(other.network.named_parameters())[name][~indices]
                # add to child.
                dict(child.network.named_parameters())[name].copy_(tmp)
        return child
    
    def mutate(self, p=0.01):
        """
        This mutates this current individual. Every param is changed with probability p. 
        It randomly adds some value.

        Args:
            p (float, optional): probability. Defaults to 0.01.
        """
        diff = 0.2
        with torch.no_grad():
            for param in self.network.parameters():
                indices = torch.rand_like(param) <= p
                param.data[indices] += diff * torch.randn_like(param.data[indices])
        
    
if __name__ == "__main__":
    e1 = EvolutionaryAgent(2, 2)
    e2 = EvolutionaryAgent(2, 2)
    c = e1.crossover(e2)
    print(list(c.network.parameters()))
    c.mutate(0.5)
    print(list(c.network.parameters()))
    print(c.network(torch.tensor([0., 0.])))
        
