import os
import math
import datetime
import pickle
from simulation.main.Simulation import SingleAgentSimulation
from agent.evolutionary.BasicPop import BasicPop
from simulation.concrete.StateRepresentations import StateRepSectorsWithDistAndVelocity
import numpy as np

if __name__ == "__main__":
    d = 'pickles/experiment1/evolutionary/proper_v1'
    if not os.path.exists(d):
        os.makedirs(d)
    
    e = BasicPop(150)
    def save(i):
        name = f'{d}/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_{e.best_fit}_gen_{i}_step_{e.num_steps * e.pop_count * i}.p'
        print(f'saving to {name}')
        with open(name, 'wb+') as f:
            pickle.dump({'agent': e.best_agent, 'simul': e.simulation}, f)
    # num_gens = math.ceil(1e6 / e.pop_count / e.num_steps)
    num_gens = 100
    np.random.seed(1234)
    for i in range(num_gens):
        simul = SingleAgentSimulation.basic_simul(StateRepSectorsWithDistAndVelocity())
        save(i)
        e.one_gen(simul)
        if i % 10 == 0 or i == num_gens - 1:
            name = f'{d}/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_whole_pop_gen_{i}_step_{e.num_steps * e.pop_count * i}.p'
            print(f'saving population to {name}')
            with open(name, 'wb+') as f:
                pickle.dump({'pop': e}, f)
    save(i+1)
    
    pass