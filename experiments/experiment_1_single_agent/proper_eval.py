import os
import glob
import pickle
from simulation.main.Simulation import SingleAgentSimulation
from simulation.concrete.HardCodeAgentDistanceSectors import HardCodeAgentDistanceSectors
from simulation.concrete.RandomAgent import RandomAgent
from simulation.concrete.RandomAgent2 import RandomAgentDumb
from experiments.eval import eval_on_special_levels
from agent.rl.external.StableBaselinesWrapper import StableBaselines
import datetime
import sys

def get_date():
    x = datetime.datetime.now()

    return (x.strftime("%Y-%m-%d_%H-%M-%S"))


def get_q_learning_names_agents():
    begin = './pickles/experiment1/SimpleQ/proper_v1/'

    files = glob.glob(begin + "*.p")
    files = sorted(files, key=os.path.getctime)


    names = []
    pickles = []
    for i in range(len(files)):
        # eps = 4000 * i * 5;
        filename = files[i]
        eps = filename.split("_")[-1].split(".p")[0]
        names.append(f"SimpleQ Steps {eps} ({filename.split('/')[-1]})")
        pickles.append(filename)
    agents = []
    print("Q Learning ", len(names), len(pickles))
    print(names)
    print(pickles);
    print("\n")
    for p in pickles:
        with open(p, 'rb') as f:
            agents.append(pickle.load(f)['agent'])
            agents[-1].learn = False
    # exit()
    return names, agents

def get_averages(dic, dic_foods):
    ans1 = {}
    ans2 = {}
    for k, v in dic.items():
        ans1[k] = sum(v) / len(v) / SingleAgentSimulation.OOB_REWARD
    
    for k, v in dic_foods.items():
        ans2[k] = sum(v) / len(v)
    return ans1, ans2

def get_dqn_names_agents():
    begin = 'pickles/experiment1/dqn/proper_v1/models/'
    names = []
    pickles = []
    # 1000 000
    
    files = glob.glob(begin + "*.zip")
    files = sorted(files, key=os.path.getctime)

    names = []
    pickles = []

    for i in range(len(files)):
        # eps = 4000 * i * 5;
        filename = files[i]
        eps = filename.split("/")[-1].split("_")[0]
        names.append(f"DQN Steps {eps} ({filename.split('/')[-1]})")
        pickles.append(filename)

    
    print("DQN", len(names))
    print(names)
    print(pickles)
    agents = []
    for p in pickles:
        # print("p = ", p)
        agents.append(StableBaselines(p, False))
    # exit()
    return names, agents

def get_evol_names_agents():
    # pickles/experiment1/evolutionary/proper_v1/04-30-2021_05-02-01_16_gen_65_step_39000000.p
    begin = 'pickles/experiment1/evolutionary/proper_v1/'
    names = []
    pickles = []
    # 1000 000
    
    files = glob.glob(begin + "*.p")
    files = sorted(files, key=os.path.getctime)
    for i in range(len(files)-1, -1, -1):
        if 'whole_pop' in files[i]:
            files.pop(i)
    names = ["Evol Start"]
    pickles = []

    # num_to_add = 10
    # length = (len(files)-2) // num_to_add
    for i in range(len(files)):
        # eps = 4000 * i * 5;
        filename = files[i]
        if 'gen_0' in filename: continue
        eps = filename.split("/")[-1].split("_")[-1].split(".p")[0]
        names.append(f"Evol {eps} ({filename.split('/')[-1]})")
        pickles.append(filename)

    agents = []
    with open(f"{begin}04-28-2021_15-47-11_whole_pop_gen_0_step_0.p", 'rb') as f:
        agents.append(pickle.load(f)['pop'].population[0])
    print("Evolutionary!", len(names), len(pickles))
    print(names)
    print(pickles)
    for p in pickles:
        # print("p = ", p)
        with open(p, 'rb') as f:
            s = pickle.load(f)
            print("Keys = ", s.keys())
            agents.append(s['agent'])
            if agents[-1] is None:
                print("IS none"); return
    # exit()
    print("Length = ", len(names), len(agents))
    return names, agents

def get_ppo_names_agents():
    begin = 'pickles/experiment1/ppo/proper_v1/models/'
    names = []
    pickles = []
    # 1000 000
    
    files = glob.glob(begin + "*.zip")
    files = sorted(files, key=os.path.getctime)

    names = []
    pickles = []

    for i in range(len(files)):
        # eps = 4000 * i * 5;
        filename = files[i]
        eps = filename.split("/")[-1].split("_")[0]
        names.append(f"PPO Steps {eps} ({filename.split('/')[-1]})")
        pickles.append(filename)

    
    print("PPO", len(names), len(pickles))
    print(names)
    print(pickles)
    agents = []
    for p in pickles:
        # print("p = ", p)
        agents.append(StableBaselines(p, True))
    # exit()
    return names, agents

def get_random_1_names_agents():
    return ["Random"], [RandomAgent()]

def get_random_2_names_agents():
    return ["RandomDumb"], [RandomAgentDumb()]

def get_hardcode_names_agent():
    return ["HardCodeAgentDistanceSectors"], [HardCodeAgentDistanceSectors()]

def eval_single(names, agents):
    little_dict = {}
    for i, (name, agent) in enumerate(zip(names, agents)):
        print(f"\tName = {name}. {i+1}/{len(names)}")
        # answer = eval_on_special_levels(agent, 2, 4000)
        answer = eval_on_special_levels(agent, 5, 10000)
        answer2 = get_averages(*answer)
        print('\t\tObst hit: ', answer2[0])
        print('\t\tFood col: ', answer2[1])
        little_dict[name] = (answer, answer2)
        print("")
    return little_dict

def eval_all():
    ANS = {}
    all_data = [get_q_learning_names_agents(), 
                get_random_1_names_agents(), 
                get_random_2_names_agents(), get_hardcode_names_agent(),
                get_ppo_names_agents(), get_dqn_names_agents()]
    all_names = ["Simple Q", "Random", "RandomDumb", "HardCode", "PPO", "DQN"]
    for N, D in zip(all_names, all_data):
        print("\n" + "==" * 100)
        print("==" * 100)
        print(f"Category {N}")
        print("==" * 100)
        names, agents = D
        little_dict = eval_single(names, agents)
        ANS[N] = little_dict
    os.makedirs('./pickles/experiment1/eval/', exist_ok=True)
    print(ANS)
    with open(f"./pickles/experiment1/eval/eval_1_{get_date()}.p", 'wb+') as f:
        pickle.dump(ANS, f)
    return ANS
            

def singles(val):
    prefix = "10k_eval_everything_simpleq_notlearn"
    if val == 'simpleq':
        print("DOING SIMPLE Q")
        ans = eval_single(*get_q_learning_names_agents())
        name = "SimpleQ"
        ANS = {}
        ANS["Simple Q"] = ans
        print(ans)
        with open(f"./pickles/experiment1/eval/proper_v1/{prefix}_{name}_{get_date()}.p", 'wb+') as f:
            pickle.dump(ANS, f)
        pass
    if val == 'ppo':
        print("DOING PPO")
        ans = eval_single(*get_ppo_names_agents())
        name = "PPO"
        ANS = {}
        ANS[name] = ans
        print(ans)
        with open(f"./pickles/experiment1/eval/proper_v1/{prefix}_{name}_{get_date()}.p", 'wb+') as f:
            pickle.dump(ANS, f)

    if val == 'dqn':
        name = "DQN"
        print("DOING DQN")
        ans = eval_single(*get_dqn_names_agents())
        ANS = {}
        ANS[name] = ans
        print(ans)
        with open(f"./pickles/experiment1/eval/proper_v1/{prefix}_{name}_{get_date()}.p", 'wb+') as f:
            pickle.dump(ANS, f)
    if val == 'nonl':
        print("DOING NON LEARNING")
        ANS = {}
        all_data = [get_random_1_names_agents(), 
                    get_random_2_names_agents(), get_hardcode_names_agent()]
        all_names = ["Random", "RandomDumb", "HardCode"]
        for N, D in zip(all_names, all_data):
            print("\n" + "==" * 100)
            print("==" * 100)
            print(f"Category {N}")
            print("==" * 100)
            names, agents = D
            little_dict = eval_single(names, agents)
            ANS[N] = little_dict
        os.makedirs('./pickles/experiment1/eval/proper_v1', exist_ok=True)
        print(ANS)
        with open(f"./pickles/experiment1/eval/proper_v1/10k_eval_1_rando12_hardcode_{get_date()}.p", 'wb+') as f:
            pickle.dump(ANS, f)
        return ANS
    if val == 'step0':
        print("DOING Those at step 0")
        ANS = {}
        all_data = [(['PPO 0 steps'], [StableBaselines("./pickles/experiment1/ppo/proper_v1/at_0_steps", True)]), 
                    (['DQN 0 steps'], [StableBaselines("./pickles/experiment1/dqn/proper_v1/at_0_steps", False)])
                    ]
        all_names = ["PPO", "DQN"]
        for N, D in zip(all_names, all_data):
            print("\n" + "==" * 100)
            print("==" * 100)
            print(f"Category {N}")
            print("==" * 100)
            names, agents = D
            little_dict = eval_single(names, agents)
            ANS[N] = little_dict
        os.makedirs('./pickles/experiment1/eval/proper_v1', exist_ok=True)
        print(ANS)
        with open(f"./pickles/experiment1/eval/proper_v1/10k_eval_1_ppo_and_dqn_0_{get_date()}.p", 'wb+') as f:
            pickle.dump(ANS, f)
        return ANS
    if val == 'ga':
        name = "GeneticAlg"
        print("DOING Genetic Alg")
        ans = eval_single(*get_evol_names_agents())
        ANS = {}
        ANS[name] = ans
        print(ans)
        with open(f"./pickles/experiment1/eval/proper_v1/{prefix}_{name}_{get_date()}.p", 'wb+') as f:
            pickle.dump(ANS, f)
def main():
    print(sys.argv[-1])
    singles(sys.argv[-1])
    # eval_all()



if __name__ == "__main__":
    main()