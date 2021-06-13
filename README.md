# EcosystemSimulation

This is for COMS4030-ACML. Simulating an ecosystem.
- Michael Beukman
- Jesse Bristow
- Thabo Ranamane

# Description
This project dealt with simulating a simple ecosystem. This ecosystem consisted of prey, food and predators. Predators can eat prey, and prey can eat food. Every entity can breed and reproduce when they have enough energy. Energy is obtained through eating another entity.

The agents observe the world in a radius around them.

The single-agent simulation is simple in that there are no breeding, dying or energy mechanics. There is also just a single prey with lots of food. We use this as basically an OpenAi Gym environment to train and evaluate reinforcement learning agents and genetic algorithms.

The multi agent simulation consists of all of the above mechanics, and there are both prey and predators. Due to computational limitations, this was reimplemented in a discrete way, so entities are in one cell at a time and their observation distance is one cell.

We test and benchmark linear Q-learning, a simple genetic algorithm and Evolutionary reinforcement learning (ERL) on the above domains.

We also optimised many of the parameters of the ERL agent, like learning rate, population size, training time, etc.

# Getting Started
Use the `env.yml` file to create your environment.
Something like:
```
conda env create -f env.yml
conda activate ecosystem
```

To actually run the code, simply run 
```
./run.sh main.py OPTION
```
Where `OPTION` is either `discrete_multi` or `single_agent`. The former runs the visualisations for the discrete multiple agent simulation and the latter runs a single agent simulation.

If you are interested in running the experiments in a headless fashion, have a look at `experiments/experiment_3_better/do_training.py`, as that runs the actual experiments, given some parameters.

The bulk of the multi-agent experiments is in `experiments/experiment_multi_agent/proper_exps.py`
# Features
- Single Agent Simulation
- Multiple Agent Simulation
- Visualisations
- Learning code for
  - Genetic Algorithm
  - Linear Q learning
  - Evolutionary Reinforcement Learning
- Experiments / Experiment Code.
- Report
# Structure
The file structure is as follows.
The main agent/learning code is inside `agent/`
```
├── agent                               -> Main Agent directory
│   ├── erl                             -> ERL Agent
│   │   ├── ErlPopulation.py            -> Population, runs ERL process
│   │   ├── SimpleQERL.py               -> Single ERL agent
│   ├── evolutionary
│   │   ├── BasicPop.py             
│   │   ├── EvolutionaryAgent.py        -> Single GA agent
│   ├── multiagent
│   │   ├── ERLMulti.py
│   └── rl
│       ├── external                    -> Unused interface with stable baselines.
│       │   ├── StableBaselinesWrapper.py
│       │   └── dqn.py
│       └── simple_q_learning.py        -> Q learning implementation
├── env.yml                             -> Environment file to reproduce our environment
├── experiments                         -> This contains a lot of code that ran our experiments.
│   ├── experiment_3_better             ->  Main Runner code for most experiments
│   │   ├── do_evaluation.py            -> This evaluates the trained agents.
│   │   ├── do_training.py              -> This runs all of our experiments, and the parameters affect what exactly it runs.
│   │   └── specific_agents_train       -> Contains code to train each specific agent type, to keep do_training.py clean.
│   │       ├── PPO.py
│   │       ├── evolutionary.py
│   │       ├── simpleq.py
│   │       └── stable_base_exp3.py
│   ├── experiment_4                    -> Population Size
│   ├── experiment_5                    -> Hyperparameters
│   ├── experiment_6                    -> Bias vs not
│   └── experiment_multi_agent          -> Multi agent sim
├── main.py                             -> Run this to see the simulation in action
├── notebooks
│   ├── DataViz.ipynb
│   └── gifs
├── pickles                             -> Where we stored most of our results. This was quite large, so we basically emptied it.
├── run.sh                              -> script to run the code. Use it like ./run.sh main.py <ARGS>
├── runAll.sh
├── shapes.py                           -> Potential patch to pyglet to work with the latest gym.
├── simulation                          -> Contains the simulation code.
│   ├── base                            -> Some interfaces.
│   ├── concrete
│   ├── discrete_ma_sim                 -> Multi agent sim to use.
│   ├── main
│   │   ├── Simulation.py               -> The main single agent simulation
│   │   ├── SingleAgentSimGym.py        -> Gym wrapper.
│   ├── multi_agent_sim                 -> Multi agent sim. DON'T USE. Rather use discrete_ma_sim. This is just here for completeness
│   └── utils                           -> Some geometry utils.
└── viz                                 -> Contains visualisation code. Rather use main.py
    ├── BadViz.py                       -> Single Agent
    ├── BadVizMultiAgent.py
    ├── DiscreteViz.py                  -> Multiple agent.
```