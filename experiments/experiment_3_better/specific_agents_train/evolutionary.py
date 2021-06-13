import os
import math
import datetime
import pickle
from simulation.base.Agent import Agent
from typing import Tuple, Union
from simulation.main.Simulation import SingleAgentSimulation
from simulation.concrete.StateRepresentations import StateRepSectorsWithDistAndVelocity
import numpy as np
from agent.erl.ErlPopulation import ErlPopulation
from agent.evolutionary.BasicPop import BasicPop

def proper_train(pop: Union[ErlPopulation, BasicPop], name, train_model_dir, num_steps) -> Tuple[Agent, np.ndarray]:
    """
        Trains either ERL or GA with num_steps and saves inside the `train_model_dir`.
        It returns a tuple consisting of the final agent, as well as an array of training rewards per episode.
    """
    d = train_model_dir
    if not os.path.exists(d):
        os.makedirs(d)
    e = pop
    def save(i):
        fname = f'{d}/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_{name}_{e.best_fit}_gen_{i}_step_{e.num_steps * e.pop_count * i}.p'
        print(f'saving to {fname}')
        with open(fname, 'wb+') as f:
            pickle.dump({'agent': e.best_agent, 'simul': e.simulation}, f)
    num_gens = math.ceil(num_steps / e.pop_count / e.num_steps)
    rewards = []
    print(f"Number of Generations: {num_gens}")
    for i in range(num_gens):
        simul = SingleAgentSimulation.basic_simul(StateRepSectorsWithDistAndVelocity())
        save(i)
        e.one_gen(simul)
        rewards.append(e.best_fit)
        if (i % 10 == 0 or i == num_gens - 1):
            fname = f'{d}/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_{name}_whole_pop_gen_{i}_step_{e.num_steps * e.pop_count * i}.p'
            print(f'saving population to {fname}')
            with open(fname, 'wb+') as f:
                pickle.dump({'pop': e}, f)
    with open(os.path.join(d, f"steps_{name}_final_{e.num_steps * e.pop_count * num_gens}.p"), 'wb+') as f:

        pickle.dump({
            'all_rewards': rewards,
            'final_agent': e.best_agent
        }, f)
    return e.best_agent, rewards