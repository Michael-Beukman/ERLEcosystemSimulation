import copy
import datetime
import glob
import math
import os
import pickle
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
from agent.rl.simple_q_learning import SimpleQLearningAgent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.concrete.StateRepresentations import (
    StateRepSectorsWithDistAndVelocity)
from simulation.main.Simulation import SimulationRunner, SingleAgentSimulation


def proper_train(num_steps):
    myd = 'pickles/experiment1/SimpleQ/proper_v1/'
    if not os.path.exists(myd):
        os.makedirs(myd)
    n = f'{myd}/RL_Q_{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_'
    sim = SingleAgentSimulation.basic_simul(StateRepSectorsWithDistAndVelocity(), num_food=200, num_obs=10)
    agent = SimpleQLearningAgent(sim.state_rep.get_num_features(), len(SimulDiscreteAction))
    s = SimulationRunner(sim, agent)
    num_steps_per_episode = 4000
    episodes = math.ceil(num_steps / num_steps_per_episode)
    print(episodes)

    gym.envs.register(
            id='SingleAgentSim-v0',
            entry_point='simulation.main.SingleAgentSimGym:SingleAgentSimGym',
            max_episode_steps=4000,
    )
    env = gym.make('SingleAgentSim-v0')
    ep_rewards = []
    for ep in range(episodes):
        agent.eligibility_trace *= 0 # reset elig before every ep.
        state = env.reset()
        done = False
        epr = 0
        while not done:
            s = State(state)
            action = agent.get_action(s)
            next_state, reward, done, _ = env.step(action.value)
            ns = State(next_state)
            agent.add_sample(s, action, reward, ns)
            state = next_state
            epr += reward
        ep_rewards.append(epr)
        if (ep) % 20 == 0:
            print(f"At episode {ep} / {episodes}, the average reward over 20 = {np.mean(ep_rewards[-20:])}")
        if (ep+1) % 100 == 0:
            print(f"At episode {ep} / {episodes}, the average reward over 100 = {np.mean(ep_rewards[-100:])}")
        # at the end of every episode, save
        if ep % 100 == 0:
            with open(n+f"steps_{ep * num_steps_per_episode}.p", 'wb+') as f:
                # this takes up a lot of storage...
                a = copy.deepcopy(agent)
                a.deltas = []
                a.rewards = []
                pickle.dump({'agent': a}, f)

if __name__ == "__main__":
    proper_train(1e7)
