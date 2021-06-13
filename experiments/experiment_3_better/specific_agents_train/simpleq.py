import math
import copy
import os
import pickle
from simulation.base.State import State
from typing import Dict, Tuple
from simulation.main.Simulation import SingleAgentSimulation
from simulation.concrete.StateRepresentations import StateRepSectorsWithDistAndVelocity
import numpy as np
from agent.rl.simple_q_learning import SimpleQLearningAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
import datetime
import gym
from simulation.base.Agent import Agent


def proper_train(train_model_dir, num_steps, args: Dict[str, float]) -> Tuple[Agent, np.ndarray]:
    """
        Trains Q learning with num_steps and saves inside the `train_model_dir`.
        It returns a tuple consisting of the final agent, as well as an array of training rewards per episode.
    """
    myd = train_model_dir
    if not os.path.exists(myd):
        os.makedirs(myd)

    n = f'{myd}/RL_Q_{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_'
    sim = SingleAgentSimulation.basic_simul(StateRepSectorsWithDistAndVelocity())
    agent = SimpleQLearningAgent(sim.state_rep.get_num_features(), len(SimulDiscreteAction), **args)
    num_steps_per_episode = 4000
    episodes = math.ceil(num_steps / num_steps_per_episode)
    print(episodes)
    env = gym.make('SingleAgentSim-v0')
    ep_rewards = []

    def save(ep, agent):
        with open(n+f"steps_{ep * num_steps_per_episode}.p", 'wb+') as f:
            # this takes up a lot of storage...
            a = copy.deepcopy(agent)
            a.deltas = []
            a.rewards = []
            pickle.dump({'agent': a}, f)

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
            save(ep, agent)
    ep_rewards = np.array(ep_rewards)
    with open(n+f"steps_final_{num_steps}.p", 'wb+') as f:
        a = copy.deepcopy(agent)
        a.deltas = []
        a.rewards = []
        pickle.dump({
            'all_rewards': ep_rewards,
            'final_agent': a
        }, f)

    return agent, ep_rewards

if __name__ == "__main__":
    proper_train(1e7)