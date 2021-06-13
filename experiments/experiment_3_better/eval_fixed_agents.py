import copy
import os
import pickle
from simulation.base.Agent import Agent
import experiments.experiment_3_better.specific_agents_train.simpleq as SimpleQ
import experiments.experiment_3_better.specific_agents_train.evolutionary as EvolTrainer
import experiments.experiment_3_better.specific_agents_train.stable_base_exp3 as PPO

from agent.erl.ErlPopulation import ErlPopulation
from agent.evolutionary.BasicPop import BasicPop
from experiments.experiment_3_better.do_evaluation import evaluate_agent
import fire
import gym
import numpy as np
import datetime
from simulation.concrete.RandomAgent2 import RandomAgentDumb
from simulation.concrete.HardCodeAgentDistanceSectors import HardCodeAgentDistanceSectors
gym.envs.register(
    id='SingleAgentSim-v0',
    entry_point='simulation.main.SingleAgentSimGym:SingleAgentSimGym',
    max_episode_steps=4000,
)
# todo maybe params!! (like alpha, gamma, etc). So we can do that now and not have to do it later.


def get_date():
    return datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
def do_eval_fixed(
    seed: int, agent_name: str
):
    np.random.seed(seed + 1000)
    # todo change to proper amounts.
    num_steps = 1e7
    POP_SIZE = 150
    EVAL_RUNS = 100
    EVAL_STEPS = 4000

    base = f"pickles/experiment_3/proper_v3/{seed}/{agent_name}/{num_steps}/{get_date()}"
    train_model_dir = os.path.join(base, "models")
    metrics_dir = os.path.join(base, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    if agent_name == 'Random':
        agent = RandomAgentDumb()
    elif agent_name == 'HardCode':
        agent = HardCodeAgentDistanceSectors()
    else: 
        print(f"{agent_name} not valid")
        exit()

    print(f"Starting eval of agent {agent_name} with")
    temp_agent = copy.deepcopy(agent)
    results = evaluate_agent(temp_agent, num_runs = EVAL_RUNS, num_steps_per_run = EVAL_STEPS)
    print("Results = ", results)
    with open(os.path.join(metrics_dir, f"eval_rewards_eps_0.p"), 'wb+') as f:
        pickle.dump({
            'eval_counts': results
        }, f)


if __name__ == "__main__":
    fire.Fire(do_eval_fixed)