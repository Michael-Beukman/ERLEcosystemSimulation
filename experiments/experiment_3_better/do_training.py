import copy
import glob
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
gym.envs.register(
    id='SingleAgentSim-v0',
    entry_point='simulation.main.SingleAgentSimGym:SingleAgentSimGym',
    max_episode_steps=4000,
)

def get_date():
    return datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
def train(
    seed: int, method: str,
    
    POP_SIZE=50,
    alpha= 0.005, gamma= 0.99, lambd= 0.9,
    eps_init= 0.1, eps_min= 0.02, eps_decay= 0.999
):
    """
    This runs training for a single agent with a single seed and evaluates it too.
    Results get stored in pickles/experiment_6/proper_v6/{seed}/{method}/{num_steps}/{POP_SIZE}/{arg_string}/{get_date()}
    Usage like:
        ./run.sh experiments/experiment_3_better/do_training.py 0 ERL --POP_SIZE 50 --alpha 0.0005 --gamma 0.99 --lambd 0.9 2>&1 | tee $log_name"_0.log"
    for a single invocation, but ideally you'd run multiple seeds, like this:
        ./run.sh experiments/experiment_3_better/do_training.py 0 ERL --POP_SIZE 50 --alpha 0.0005 --gamma 0.99 --lambd 0.9 2>&1 | tee $log_name"_0.log"
        ./run.sh experiments/experiment_3_better/do_training.py 1 ERL --POP_SIZE 50 --alpha 0.0005 --gamma 0.99 --lambd 0.9 2>&1 | tee $log_name"_1.log" &
        ./run.sh experiments/experiment_3_better/do_training.py 2 ERL --POP_SIZE 50 --alpha 0.0005 --gamma 0.99 --lambd 0.9 2>&1 | tee $log_name"_2.log" &

    Args:
        seed (int): Random seed
        method (str): ERL, SimpleQ, GA, or PPO
        POP_SIZE (int, optional): Population size of ERL. Defaults to 50.

        Some
        alpha (float, optional) Defaults to 0.005.
        gamma (float, optional)  Defaults to 0.99.
        lambd (float, optional) Defaults to 0.9.
        eps_init (float, optional) Defaults to 0.1.
        eps_min (float, optional): epsilon doesn't go below this value. Defaults to 0.02.
        eps_decay (float, optional): Every step decays epsilon with this value Defaults to 0.999.
    """
    np.random.seed(seed + 1000)
    # todo change to proper amounts.
    num_steps = 1e7
    # POP_SIZE = POP_SIZE
    EVAL_RUNS = 100
    EVAL_STEPS = 4000
    args = {
        'alpha': alpha, 'gamma': gamma, 'lambd': lambd,
        'eps_init': eps_init, 'eps_min': eps_min, 'eps_decay': eps_decay
    }
    arg_string = "__".join(
        [f"{k}_{v}" for k, v in args.items()]
    )
    print("ARGS STRING LENGTH", len(arg_string), arg_string)
    print("ARGS= ", args)
    # exit()
    base = f"pickles/experiment_6/proper_v6/{seed}/{method}/{num_steps}/{POP_SIZE}/{arg_string}/{get_date()}"
    train_model_dir = os.path.join(base, "models")
    metrics_dir = os.path.join(base, "metrics")

    # print("Args = ", args)
    # exit()
    print("POP SIZE = ", POP_SIZE)
    print("EVAL RUNS = ", EVAL_RUNS)
    print("EVAL STEPS = ", EVAL_STEPS)
    print(f"Train model dir = {train_model_dir}")
    os.makedirs(train_model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    if method == "PPO":
        agent, train_rewards = PPO.proper_train(
            "ppo", train_model_dir, num_steps)
    elif method == "SimpleQ":
        agent, train_rewards = SimpleQ.proper_train(train_model_dir, num_steps, args=args)
    elif method == "GA":
        agent, train_rewards = EvolTrainer.proper_train(
            BasicPop(POP_SIZE, num_steps=4000), "GA", train_model_dir, num_steps)
    elif method == "ERL":
        pop = ErlPopulation(POP_SIZE, args = args, num_steps=4000)
        print("POPULATION GEN COUNT = ", pop.gen_count)
        print("Feature size", pop.population[0].weights.shape)
        agent, train_rewards = EvolTrainer.proper_train(
            pop, "ERL", train_model_dir, 2e6)
    else:
        print(f"Method {method} is invalid. Exiting"); exit()
    
    # save train rewards
    with open(os.path.join(metrics_dir, "train_rewards.p"), 'wb+') as f:
        pickle.dump({
            'training_rewards': train_rewards,
            'method': method,   
            'args': args
        }, f)

    # Now we need to do evaluation.

    # can I end the loop after one epsilon? (e.g. GA, PPO that don't have epsilon)
    can_stop = False
    for eps in [0, 0.05]:
        print(f"Starting eval with eps = {eps}")
        temp_agent = copy.deepcopy(agent)
        if hasattr(temp_agent, 'learn'):
            temp_agent.learn = False
        else:
            print(f"{method} does not have learn attribute")
        if hasattr(temp_agent, 'epsilon'):
            temp_agent.epsilon = eps
            temp_agent.eps_min = eps
            temp_agent.eps_decay = 1
        else:
            can_stop = True
        results = evaluate_agent(temp_agent, num_runs = EVAL_RUNS, num_steps_per_run = EVAL_STEPS)
        print("Results = ", results)
        with open(os.path.join(metrics_dir, f"eval_rewards_eps_{eps}.p"), 'wb+') as f:
            pickle.dump({
                'eval_counts': results,
                'method': method,
                'args': args,
                'epsilon': eps
            }, f)
        if can_stop:
            break



if __name__ == "__main__":
    fire.Fire(train)