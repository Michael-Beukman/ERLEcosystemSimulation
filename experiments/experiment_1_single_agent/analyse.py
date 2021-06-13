import glob
import pickle
from simulation.main.Simulation import SingleAgentSimulation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def get_num_steps_trained(k):
    if "Random" in k or "Hard" in k: return 0
    print("K = ", k)
    if '100k' in k: return 1e5
    if '1e7' in k: return 1e7
    if '0 steps' in k: return 0
    if 'Evol Start' in k: return 0
    if 'Evol End' in k: return 64 * 4e3

    if 'ERL start' in k: return 0
    # if 'ERL end' in k: return 64 * 4e3

    if 'Episode 0' in k: return 0
    if "Evol" == k[:4]:
        steps = int(k.split(' ')[1])#/ 150
        return steps
    if "ERL" == k[:3]:
        print(k.split(' '))
        steps = int(k.split(' ')[2])#/ 150
        return steps
    if "SimpleQ" in k or "RL_Q" in k:
        return int(k.split('_')[-1].split(".p")[0])
    steps = int(k.split('(')[-1].split("_")[0])
    return steps
    # return 1;

def get_value(vals, metric, level):
    tmp, ave = vals
    print("==" * 10)
    print(tmp)
    print("==" * 10 + "\n")

    # Test thing.
    new_ave_obs = {}
    new_ave_food = {}
    for key in tmp[0]:
        li = tmp[0][key]
        li = sorted(li)[1:-1]
        new_ave_obs[key] = np.mean(li) / SingleAgentSimulation.OOB_REWARD
    for key in tmp[1]:
        li = tmp[1][key]
        li = sorted(li)[1:-1]
        new_ave_food[key] = np.mean(li)
    print("\tOld = ", ave)
    print("\tNew = ", (new_ave_obs, new_ave_food))
    ave = (new_ave_obs, new_ave_food)
    obstacles, food = ave
    if metric == 'foodc' or metric == 'food':
        thing = food
    elif metric == 'obst':
        thing = obstacles
    elif metric == 'sum':
        thing = {}
        for k in obstacles:
            thing[k] = food[k] - obstacles[k]
    elif metric == 'reward':
        thing = {}
        for k in obstacles:
            thing[k] = food[k] - 100 * obstacles[k]
    
    return thing[level]

maxi = 5
# either GEN, RL, or ALL.
mode = "GEN"

def new_one():
    global maxi

    all_dic = {}
    dir = '4k'
    if dir == '10k':
        files = glob.glob("pickles/experiment1/eval/proper_v1/10k_eval_everything*.p") + \
            glob.glob("pickles/experiment1/eval/proper_v1/10k_eval_1_rando12_*p")
    else:
        files = glob.glob("pickles/experiment1/eval/proper_v1/4k_eval_everything*.p") + \
            glob.glob("pickles/experiment1/eval/proper_v1/eval_1_rando12_*p")
    files = [f for f in files if '_SimpleQ_' not in f or ('_SimpleQ_' in f and ('not_learning' in f or 'notlearn' in f))]
    # print(files); exit()
    for f in files:
        with open(f, 'rb') as file:
            d = pickle.load(file)
            # print("F = ", f, d.keys(), )
            # if ('PPO') in d:print('hey', d['PPO'].keys())
            for key in d:
                if key in all_dic:
                    for kkk in d[key]:
                        all_dic[key][kkk] = d[key][kkk]
                else:
                    all_dic[key] = d[key]
    # print(all_dic.keys());
    # print(all_dic['ERL'].keys()); exit()
    def do_specific_thing(metric, level, ax):
        global maxi, mode
        for i in all_dic:
            if i in ['Random', 'DQN']: # , "PPO", 'DQN'
                continue
            if mode == 'GEN':
                maxi = 3.5e7
                if i in ['PPO', 'Simple Q'] : 
                    continue
            elif mode == 'RL':
                maxi = 1e7
                if i in ['ERL', 'GeneticAlg'] : 
                    continue
            else:
                maxi = 1e7
                if i in ['PPO'] : 
                    continue
                print("Mode is ALL")
            # print("i ", i)
            thingy_list = []
            for k in all_dic[i]:
                # print("\t", k)
                # print("\t\t", all_dic[i][k][0][0].keys())
                num_steps = get_num_steps_trained(k)
                average_value = get_value(all_dic[i][k], metric, level)
                if num_steps <= maxi:
                    thingy_list.append((num_steps, average_value))
            thingy_list = sorted(thingy_list)
            num_steps, vals = map(list,zip(*thingy_list))
            if len(num_steps) == 1:
                # num_steps.append(1e7)
                num_steps.append(maxi)
                vals.append(vals[0])
            # print(num_steps)
            # if 1:num_steps[-1] = 1e7
            if False and (i == "GeneticAlg" or i == "ERL"):
                # print("In genetic alg")
                maxi = num_steps[-1]
                for l in range(len(num_steps)):
                    num_steps[l] = (num_steps[l] / maxi) * 1e7

            # ax.plot(num_steps, vals, label=i)
            # vals = np.clip(vals, -100, 100)
            # v = savgol_filter(vals, 3, 2) if len(vals) > 2 else vals
            v = vals
            # v = vals
            # v = np.clip(v, -100, 100)
            # print("SHAPE", v.shape)
            ax.plot(num_steps, v, label=i)
            print("Max = ", max(num_steps))
            # ax.set_ylim(-100, 100)
            if (metric == 'obst' or 1) and True:
                ax.set_yscale("symlog")
            else:
                ax.set_yscale("linear")
    
    levels = ['obstacles', 'foodhunt', 'combination', 'fixed_random']
    for m in ['food', 'obst', 'sum', 'reward']:
        fig, axs = plt.subplots(2, 2, figsize=(20,20))
        axs = axs.ravel()
        for level, ax in zip(levels, axs):
            do_specific_thing(m, level, ax)
            ax.set_title(f"Metric = {m}. Level = {level}")
            ax.legend()
        # plt.show()
        plt.close()
        fig.savefig(f'./plots/experiment_1/proper_v1/{dir}/metric_{m}_mode_{mode}.png')
# 9 920 000
def analyse_training_reward():
    def simple_q():
        log_file = "logs/experiment_1/simple_q/proper_v1.log"
        with open(log_file, 'r') as f:
            s = f.readlines()
        # t = s[-1]
        s = [i for i in s if 'over 20' in i]
        # print(s); exit()
        steps = [int(i.split("At episode ")[1].split(" ")[0]) * 4000 for i in s]
        # print(steps); exit()
        # s.append(t)
        s = list(map(float, [i.strip().split('= ')[-1] for i in s]))
        # s = savgol_filter(s, 5, 2)
        # plt.plot(s)
        # plt.show()
        return steps, s
    def ga(is_erl=False):
        if is_erl:
            log_file = 'logs/experiment_1/erl/proper_v1/0501_first_run_erl.log'
        else:
            log_file = 'logs/experiment_1/genetic/proper_v1.log'
        with open(log_file, 'r') as f:
            s = f.readlines()
        s =  [i for i in s if '....' in i and 'gen' in i]
        steps = [int(i.split("For gen")[1].split(": ")[0])*4000 * 150 for i in s]
        # print(gens); exit()
        s = [i.strip().split("For gen")[1].split(": ")[1] for i in s]
        # max
        s = list(map(float, [i.split(". Min")[0].split(" = ")[1] for i in s]))
        # print(s)
        # plt.plot(s)
        # plt.show()
        return steps,s
        pass
    def ppo_and_dqn(mode='ppo'):
        if mode == 'ppo':
            log_file = 'logs/experiment_1/ppo/proper_v1.log'
        else:
            log_file = 'logs/experiment_1/dqn/proper_v1.log'
        with open(log_file, 'r') as f:
            s = f.readlines()
        steps =  [i for i in s if 'total_timesteps' in i]
        s =  [i for i in s if 'ep_rew_mean' in i]
        s = list(map(float, [i.strip()[1:-1].split("| ")[1].strip() for i in s]))
        steps = list(map(float, [i.strip()[1:-1].split("| ")[1].strip() for i in steps]))
        # print(steps)
        # plt.plot(s)
        # plt.show()
        steps = steps[:-1]
        # print("L =",len(steps), len(s)); exit()
        return steps, s
    
    v3, ppo = ppo_and_dqn('ppo')
    v1, q = simple_q()
    v2, g = ga()
    v4, erl = ga(True)
    # v1 = np.arange(len(q))
    # v2 = np.arange(len(g))
    # v3 = np.arange(len(ppo))
    # v4 = np.arange(len(erl))

    # v1 = v1 / v1.max() * 1e7
    # v2 = v2 / v2.max() * 1e7
    # v3 = v3 / v3.max() * 1e7
    # v4 = v4 / v4.max() * 1e7
    vs = [v1, v2, v3, v4]
    names = ['q learning', 'genetic alg', "PPO", 'ERL']
    vals = [q, g, ppo, erl]
    for v, n, nums in zip(vs, names, vals):
        if n =='PPO':continue
        v = np.array(v)
        idx = v <= 1e7

        plt.plot(v[idx], np.array(nums)[idx], label=n)

    # plt.plot(v1, q, label='q learning')
    # plt.plot(v2, g, label='genetic alg')
    # plt.plot(v4, erl, label='ERL')
    # plt.plot(v3, dqn, label='dqn')
    plt.title("Training reward")
    # plt.ylim(bottom=-500)
    plt.legend()
    # plt.show()
    plt.savefig("plots/experiment_1/proper_v1/training_rewards/training_reward.png")
if __name__ == "__main__":
    for mode in ['GEN', 'RL', "ALL"]:new_one()
    analyse_training_reward()
