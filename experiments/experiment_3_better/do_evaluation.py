from simulation.main.Simulation import SimulationRunner, SingleAgentSimulation
from simulation.base.Agent import Agent
import copy
from simulation.main.Entity import EntityType
import numpy as np

def get_means(counts, name):
    lifood = counts['food'][name]
    liobs = counts['obs'][name]

    return {"food": np.mean(lifood), "obs": np.mean(liobs)}

def evaluate_agent(agent: Agent, num_runs=100, num_steps_per_run=4000):
    """
        Evaluates the agent on 4 different levels, num_runs times.
        Returns a dictionary of the form
        {
            'food': {
                'obstacles': [num_food_collected_ep1, num_food_collected_ep2, num_food_collected_ep3, etc...],
                'foodhunt': [num_food_collected_ep1, num_food_collected_ep2, num_food_collected_ep3, etc...],
                etc..
            },
            'obs': {
                'obstacles': [num_obs_hit_ep1, num_obs_hit_ep2, num_obs_hit_ep3, etc...],
                'foodhunt': [num_obs_hit_ep1, num_obs_hit_ep2, num_obs_hit_ep3, etc...],
                etc..
            }
        }
    """

    names = ['obstacles', 'foodhunt', 'combination', 'fixed_random']
    counts = {
        'food' : {
            n: [] for n in names
        },
        'obs'  : {
            n: [] for n in names
        }
    }
    funcs = [SingleAgentSimulation.obstacle_avoidance_level, SingleAgentSimulation.food_finding_level, SingleAgentSimulation.combination_hard_coded_eval_level, SingleAgentSimulation.fixed_random_eval_level]
    for index, (name, func) in enumerate(zip(names, funcs)):
        print(f"Doing Eval Level {name}")
        for run_no in range(num_runs):
            print(f"\r\t\tEpisode {run_no} / {num_runs}", end='')
            if name == 'fixed_random':
                level = funcs[index](run_no + 1234567)
            else:
                level = funcs[index]()
            
            # ensure no dodgy things will happen regarding modifying the agent.
            tmp = copy.deepcopy(agent)
            s = SimulationRunner(level, tmp)
            for j in range(num_steps_per_run):
                s.update()
            num_obs_hit = (level.total_rewards[0] - level.score[0] * level.rewards[EntityType.FOOD]) / SingleAgentSimulation.OOB_REWARD
            num_food_gotten = level.score[0]
            counts['food'][name].append(num_food_gotten)
            counts['obs'][name].append(num_obs_hit)
        print(f"\n\tRewards for this step: {get_means(counts, name)}")
    return counts
    