import fire
from agent.multiagent.ERLMulti import ERLMulti
import datetime
from simulation.main.Entity import EntityType
# import pickle
import dill as pickle
import numpy as np
import os
from simulation.multi_agent_sim.hardcode_agents.HardCodeAgents import *
from simulation.multi_agent_sim.MultiStateRep import MultiAgentStateRep
# from simulation.multi_agent_sim.MultiAgentSimulation import MultiAgentSimulation
from simulation.discrete_ma_sim.discrete_multi_agent_sim import DiscreteMultiAgentSimulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import random


def main(seed: int, nerl, nrando, nhard,
npred_erl,
npred_rando,
npred_hard,
):
    np.random.seed(seed)
    random.seed(seed)

    n_foods = 100
    d = dict(
        size=64,
        # n_preds=npred, n_preys=n_preys,
        n_foods=n_foods,
        n_preds={
            'erl': npred_erl,
            'random': npred_rando,
            'hard': npred_hard
        }, n_preys={
            'erl': nerl,
            'random': nrando,
            'hard': nhard
        },
        entity_lifetimes={
            EntityType.FOOD: float('inf'),
            # float('inf'), # 4000*100,
            EntityType.PREY: 300,
            # float('inf'), #4000*100,
            EntityType.PREDATOR: 600
        },
        entity_energy_to_reproduce={
            EntityType.PREY: 1200,
            EntityType.PREDATOR: 1200
        },
    )
    all_the_data = {
        'food': [],

        'prey erl': [],
        'prey rando': [],
        'prey hard': [],

        'pred erl': [],
        'pred rando': [],
        'pred hard': [],

        'energy_pred': [],
        'energy_prey': [],
    }
    sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(**d)
    modder = 500
    NUM_STEPS = 10000

    def c(agents, clas):
        return sum([1 for i in agents.values() if isinstance(i, clas)])
    for i in range(NUM_STEPS):
        sim.update()
        if i % modder == 0 and i != 0:
            s = f"SEED {seed} At Step = {i}. Prey = {len(sim.e_prey)}. Preds = {len(sim.e_predators)}. Food = {len(sim.foods)}. FPS = {round(1/sim.dt)}. DT = {round(sim.dt * 1000)}ms"
            print(s)

        counts_prey = {
            'hard': c(sim.a_prey, HardCodePrey),
            'rando': c(sim.a_prey, RandomMultiAgent),
            'erl': c(sim.a_prey, ERLMulti),
        }

        counts_pred = {
            'hard': c(sim.a_predators, HardCodePred),
            'rando': c(sim.a_predators, RandomMultiAgent),
            'erl': c(sim.a_predators, ERLMulti),
        }
        all_the_data['prey hard'].append(counts_prey['hard'])
        all_the_data['prey rando'].append(counts_prey['rando'])
        all_the_data['prey erl'].append(counts_prey['erl'])

        all_the_data['pred hard'].append(counts_pred['hard'])
        all_the_data['pred rando'].append(counts_pred['rando'])
        all_the_data['pred erl'].append(counts_pred['erl'])

        all_the_data['food'].append(len(sim.foods))

    # npred_erl, npred_rando, npred_hard,
    def save(data):
        name = f"erl_{nerl}_rando_{nrando}_hard_{nhard}_{npred_erl}_{npred_rando}_{npred_hard}"
        d = f'./pickles/experiment_multi_agent_test/proper_exps/v6_proper_simple/preys_pred/{name}/{seed}/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'
        n = f'{d}/data.p'
        os.makedirs(d, exist_ok=True)
        with open(n, 'wb+') as f:
            pickle.dump({'data': data, 'args': d}, f)
    save(all_the_data)

def meep_main(seed):
    # print(f"SEED = {seed}"); exit()
    if 0:
        nums = [0, 5]
        for nerl in nums:
            for nrando in nums:
                for nhard in nums:
                    print(f"Seed {seed} DOING nerl={nerl}, rando={nrando}, nhard={nhard}")
                    main(seed, nerl, nrando, nhard)
    if 0:
        nums = [0, 5]
        for nerl in nums:
            for nrando in nums:
                for nhard in nums:
                    print(f"Seed {seed} DOING nerl={nerl}, rando={nrando}, nhard={nhard}")
                    main(seed, nerl=5, nrando=0, nhard=0, # prey
                                #  pred
                                npred_erl = nerl,
                                npred_rando = nrando,
                                npred_hard = nhard,
                                )
    """
    n_foods=n_foods,
        n_preds={
            'erl': 10,
            'random': 10,
            'hard': 0
        }, n_preys={
            'erl': 10,
            'random': 10,
            'hard': 10
        },
        entity_lifetimes={
            EntityType.FOOD: float('inf'),
            # float('inf'), # 4000*100,
            EntityType.PREY: 300,
            # float('inf'), #4000*100,
            EntityType.PREDATOR: 600
        },
        entity_energy_to_reproduce={
            EntityType.PREY: 1200,
            EntityType.PREDATOR: 1200
        },
    """
    main(seed, nerl=10, nrando=10, nhard=10, # prey
            #  pred
            npred_erl = 10,
            npred_rando = 10,
            npred_hard = 0,
            )

if __name__ == '__main__':
    fire.Fire(meep_main)
