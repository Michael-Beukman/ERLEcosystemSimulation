import copy
import os
import pickle
from tokenize import Single
from typing import Dict, List
from simulation.main.Simulation import SimulationRunner, SingleAgentSimulation
from agent.erl.SimpleQERL import ERLAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.concrete.StateRepresentations import StateRepSectorsWithDistAndVelocity
import numpy as np
import datetime

class ErlPopulation:
    """
    Main class for erl learning.
    This basically does reinforcement learning (q learning as in agents/rl/simple_q_learning.py) during an agent's lifetime
    and performs genetic algorithm operations to get the next generation.
    """
    def __init__(self, pop_count, args: Dict[str, float], num_steps=4000):
        """
        Args:
            pop_count (int): Number of individuals
            args (Dict[str, float]): The arguments to pass to each `ERLAgent` (e.g. alpha, gamma, lambda, etc.)
            num_steps (int, optional): Number of steps to evaluate (and learn) for per generation. Defaults to 4000.
        """
        self.simulation = SingleAgentSimulation.basic_simul(StateRepSectorsWithDistAndVelocity())
        self.pop_count = pop_count
        num_obs = self.simulation.get_num_features()
        num_actions = len(SimulDiscreteAction)
        
        self.population = [ERLAgent(num_obs, num_actions, **args) for i in range(pop_count)]

        self.gen_count = 0 
        
        self.best_agent = None
        self.best_fit = None
        self.runner = None
        self.num_steps = num_steps
        
    def one_gen(self, simul: SingleAgentSimulation):
        """
        Performs one generation consisting of evaluation and breeding.
        Args:
            new_simul (SingleAgentSimulation): The new simulation to use for evaluation.
        """
        self.simulation = simul
        probs = self.evaluate_and_learn()
        self.breed(probs)
        self.gen_count+=1
    
    def breed(self, probs: List[float]):
        """
        This performs a single iteration of breeding. 
        Args:
            probs (List[float]): The probability to choose each individual. Same length as self.population
        """
        
        # Get and store the best agent for the next generation to not get worse because of randomness.
        best_agent = np.argmax(probs)
        self.best_agent = self.population[best_agent]
        self.best_agent.eligibility_trace *= 0
        new_pop = [self.best_agent]
        
        while len(new_pop) < self.pop_count:
            # Get two parents. Higher fitnesses mean more chance of getting chosen to breed.
            a1: ERLAgent = np.random.choice(self.population, p=probs)
            a2: ERLAgent = np.random.choice(self.population, p=probs)
            c = 0
            # Try and not use the same individual twice.
            while a2 == a1 and c < 5:
                c += 1
                a2 = np.random.choice(self.population, p=probs)

            if (a1 == a2):print(" # ", end='')
            # Crossover            
            child = a1.crossover(a2)
            # Mutation
            child.mutate(0.2)
            # add to next gen.
            new_pop.append(child)
        self.population = new_pop
    
    def evaluate_and_learn(self):
        """
        This evaluates the entire population and uses rank based selection to deal with negative values.
        Each agent also learns when interacting with the env.

        Returns:
            List[float]: The probability that each individual should be chosen to breed. normalised, so sum == 1.
        """
        total_fit = 0
        max_fit = -1e12
        min_fit = 1e12
        things = []
        # For every agent
        for index, a in enumerate(self.population):
            print('.', end='', flush=True)
            tmp = copy.deepcopy(self.simulation)
            runner = SimulationRunner(tmp, a)
            # Run the simulation for 4k steps, this individual performs actions and gets reward.
            # Agent learns during this process.
            for i in range(self.num_steps):
                runner.update()
            # The fitness = total reward obtained. Thus, higher reward => higher fitness.                
            a.fitness = tmp.total_rewards[0]
            # logging
            max_fit = max(max_fit, a.fitness)
            min_fit = min(min_fit, a.fitness)
            # fitness and index
            things.append((a.fitness, index))
        for a in self.population: 
            total_fit += a.fitness
        temp = max_fit
        # Log things for progress estimates.
        print(f"For gen {self.gen_count}: Max Fit = {max_fit}. Min = {min_fit}. Average = {total_fit/(max(1, self.pop_count))}")
        self.best_fit = max_fit
        probs = [a.fitness / (total_fit if total_fit !=0 else 1) for a in self.population]
        
        # sorts so that the biggest one is at the beginning.
        things = sorted(things, reverse=True)
        """
        This now creates a list of probabilities, such that p[i] > p[i+1] and sum(p) = 1.
        This is used because we have negative fitnesses. If we didn't then just fitness / total_fitness could be the individual's
        probability of selection, but we have so we need another option.
        So we create a list (like [0.2, 0.16, 0.12, ...]) and use the individuals rank to index into this 
        Thus, the best individual will have a probability of 0.2, second would be 0.16, etc.
        """
        init = 1/5
        remaining = 1
        temp = []
        for index in range(len(things)):
            if index == len(things)-1:
                temp.append(remaining)
            else:
                temp.append(remaining * init)
            remaining = (remaining) * (1 - init)

        probs = [0 for _ in things]
        # Then just allocate probabilities according to relative rank.
        for i, (_, index) in enumerate(things):
            probs[index] = temp[i]
        return probs

        
if __name__ == "__main__":
    print('tmp')
    if not os.path.exists('pickles'):
        os.makedirs('pickles')
    
    def save(i):
        name = f'pickles/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_{e.best_fit}_{i}.p'
        print(f'saving to {name}')
        with open(name, 'wb+') as f:
            pickle.dump({'agent': e.best_agent, 'simul': e.simulation}, f)
    e = BasicPop(10)
    for i in range(2):
        if i % 50 == 0:
            save(i)
        e.one_gen()
    save(i)
    
    pass
