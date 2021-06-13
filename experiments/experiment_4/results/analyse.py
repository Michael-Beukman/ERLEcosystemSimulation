import glob
from pprint import pprint
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
def training_plots():
    data = defaultdict(lambda : [])
    for i in range(5):
        thing = f'pickles/experiment_3/proper_v3/{i}/*/10000000.0/*/metrics/train_rewards.p'
        
        g = glob.glob(thing)
        g = [i for i in g if "ERL" not in i]
        print("Len ", len(g))
        g = g + glob.glob(f'pickles/experiment_3/proper_v3/{i}/*/10000000.0/50/*/metrics/train_rewards*.p')

        print(len(g), g)
        for file in g:
            # get names:
            name = file.split(f"proper_v3/{i}/")[1].split("/")[0]
            with open(file, 'rb') as f:
                vals = pickle.load(f)
            data[name].append(vals['training_rewards'])
    print(data)
    for key, vals in data.items():
        if key == 'PPO': continue
        size = 100
        if key == 'ERL' or key == 'GA':
            size = 1
        arr = np.array(vals)    
        
        # moving average.
        arr2 = np.zeros_like(arr)
        for seed in range(arr.shape[0]):
            for ep in range(arr.shape[1]):
                
                arr2[seed, ep] = np.mean(arr[seed, max(0, ep-size): ep + 1])
        arr = arr2
        print(key)
        print("Shape", arr.shape)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        ins = np.linspace(0, 1e7, len(mean))
        plt.plot(ins, mean, label=key)
        plt.fill_between(ins, mean - std, mean + std, alpha=0.5)
    plt.title("Moving average of reward vs number of steps trained")
    plt.xlabel("Steps trained")
    plt.ylabel("Moving average reward of last 100 episodes")
    plt.legend()
    # plt.show()
    plt.savefig("experiments/experiment_3_better/results/train_plot_without_ppo.png")
    return data

def eval_table():
    data = defaultdict(lambda : [])
    for i in range(5):
        thing = f'pickles/experiment_3/proper_v3/{i}/*/10000000.0/*/metrics/eval_rewards*.p'
        g = glob.glob(thing)
        # g = [i for i in g if "ERL" not in i]
        # print("Len ", len(g))
        g = g + glob.glob(f'pickles/experiment_3/proper_v3/{i}/*/10000000.0/50/*/metrics/eval_rewards*.p')
        # print("LEN AFTER ", len(g)); exit()
        for file in g:
            # print(file); continue
            # get names:
            # print("ree", file.split(".p")[0]);exit()
            name = file.split(f"proper_v3/{i}/")[1].split("/")[0] + "_eps_" + file.split(".p")[0].split("_")[-1]
            if "ERL" in file and "/50/" not in file:
                name += "_150_17"
            elif "ERL" in file:
                name += "50_50"
            with open(file, 'rb') as f:
                vals = pickle.load(f)
            print(vals.keys())
            data[name].append(vals['eval_counts'])
    # print(data)
    all_vals = {}
    for key, vals in data.items():
        print("="*100); print("KEY = ", key); print("="*100)
        # if key == 'PPO': continue
        values = {
            'obstacles': [],
            'foodhunt': [],
            'combination': [],
            'fixed_random': []
        }
        # v = vals[0]
        for v in vals:
            food_col = v['food']
            obst_hit = v['obs']
            for level_name in food_col:
                print(level_name)
                # print(sorted(obst_hit[level_name][-20:]))
                # print(sorted(obst_hit[level_name], reverse=True))
                # print(sorted(food_col[level_name], reverse=True))
                # print((obst_hit[level_name]))
                tmp = []
                for i, v in enumerate(obst_hit[level_name]):
                    tmp.append((v,i))
                print(sorted(tmp, reverse=True))
                print("")
                lifood = sorted(food_col[level_name])[5:-5]
                liobst = sorted(obst_hit[level_name])[5:-5]
                reward = np.mean(lifood) - np.mean(liobst) * 100
                values[level_name].append(reward)
            # print(len(v), v); exit()
        # print(values); exit()
        # arr = np.array(vals)
        # pprint(vals)
        all_vals[key] = {
            # k: (np.mean(kk), np.std(kk)) for k,kk in values.items()
            k: f"{np.round(np.mean(sorted(kk)[1:-1]))} +- {np.round(np.std(sorted(kk)[1:-1]))}" for k,kk in values.items()
        }
    # print(all_vals)
    df = pd.DataFrame(all_vals)
    df = df.T.sort_values(by='fixed_random')
    df['Tmp'] = df['fixed_random'].map(lambda x: float(x.split(" ")[0]))
    df = df.sort_values(by='Tmp')
    df = df.drop(["Tmp"], axis=1)
    print(df.to_latex("experiments/experiment_3_better/results/eval_tables.tex"))
    print(df)

def get_diff_pop_pickle_names():
    vals = defaultdict(lambda : {})
    for i in range(5):
        for k in [10, 25, 100]:
            s = f'./pickles/experiment_4/proper_v4/{i}/ERL/10000000.0/{k}/*/*/metrics'
            vals[i][k] = s

        vals[i][50] = f'./pickles/experiment_3/proper_v3/{i}/ERL/10000000.0/50/*/metrics'
        vals[i][150] = f'./pickles/experiment_3/proper_v3/{i}/ERL/10000000.0/05*/metrics'
    return vals

def get_values(pickle_names, name='train', eps='0'):
    vals = defaultdict(lambda : [])
    for seed in pickle_names.keys():
        thing = pickle_names[seed]
        for size in thing.keys():
            g = glob.glob(thing[size])
            assert len(g) == 1
            filename  = g[0]
            if name == 'train':
                with open(f"{filename}/train_rewards.p", 'rb') as f:
                    data = pickle.load(f)
                    
                vals[size].append(data['training_rewards'])
            else:
                with open(f"{filename}/eval_rewards_eps_{eps}.p", 'rb') as f:
                    data = pickle.load(f)
                    
                vals[size].append(data['eval_counts'])
    return vals

def plot(vals):
    print("KEYS = ", vals.keys())
    for size, li in vals.items():
        li = np.array(li)
        print(li.shape)
        if size in [10, 25]: continue
        mean = np.mean(li, axis=0)
        std = np.std(li, axis=0)
        name = f"ERL size = {size}"
        ins = np.linspace(0, 1e7, len(mean))
        plt.plot(ins, mean, label=name)
        plt.fill_between(ins, mean - std, mean + std, alpha=0.5)
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()

def do_eval_table(vals):
    all_vals = {}
    test = {}
    for key, vals in vals.items():
        test[key] = {
            'obstacles': [],
            'foodhunt': [],
            'combination': [],
            'fixed_random': []
        }
        # print("="*100); print("KEY = ", key); print("="*100)
        # if key == 'PPO': continue
        values = {
            'obstacles': [],
            'foodhunt': [],
            'combination': [],
            'fixed_random': []
        }
        # v = vals[0]
        for v in vals:
            food_col = v['food']
            obst_hit = v['obs']
            for level_name in food_col:
                # print(level_name)
                # print(sorted(obst_hit[level_name][-20:]))
                # print(sorted(obst_hit[level_name], reverse=True))
                # print(sorted(food_col[level_name], reverse=True))
                # print((obst_hit[level_name]))
                tmp = []
                for i, v in enumerate(obst_hit[level_name]):
                    # tmp.append((v,i))
                    tmp.append((v))
                # print(sorted(tmp, reverse=True))
                def filter(li):
                    # ans2 = sorted(li)[40:-40]
                    N = -3000
                    idx = li < N
                    li[idx] *= 0
                    li[idx] += N
                    return li
                    # ans2 = sorted(li)[20:-20]
                    
                    l = li
                    l = l[(l>np.quantile(l,0.25)) & (l<np.quantile(l,0.75))].tolist()
                    # print(len(l), l)
                    return ans2
                    return l

                test[key][level_name].append(sorted(tmp, reverse=True))
                # print("")
                lireward = np.array(food_col[level_name]) - np.array(obst_hit[level_name]) * 100
                lireward = filter(lireward)
                # print("LENG = ", len(lireward))
                # lireward = [np.median(lireward)]
                # lifood = filter(food_col[level_name]) # [20:-20]
                # liobst = filter(obst_hit[level_name]) # [20:-20]
                # reward = np.mean(lifood) - np.mean(liobst) * 100
                reward = np.mean(lireward)
                values[level_name].append(reward)
            # print(len(v), v); exit()
        # print(values); exit()
        # arr = np.array(vals)
        # pprint(vals)
        all_vals[key] = {
            # k: (np.mean(kk), np.std(kk)) for k,kk in values.items()
            k: f"{np.round(np.mean(sorted(kk)[1:-1]))} +- {np.round(np.std(sorted(kk)[1:-1]))}" for k,kk in values.items()
        }
    # print(all_vals)
    df = pd.DataFrame(all_vals)
    df = df.T.sort_values(by='fixed_random')
    df['Tmp'] = df['fixed_random'].map(lambda x: float(x.split(" ")[0]))
    
    df['Tmp'] += df['obstacles'].map(lambda x: float(x.split(" ")[0]))
    df['Tmp'] += df['foodhunt'].map(lambda x: float(x.split(" ")[0]))
    df['Tmp'] += df['combination'].map(lambda x: float(x.split(" ")[0]))
    df = df.sort_values(by='Tmp')
    # df = df.drop(["Tmp"], axis=1)
    # print(df.to_latex("experiments/experiment_3_better/results/eval_tables.tex"))
    print(df)

if __name__ == "__main__":
    vs = get_diff_pop_pickle_names()
    pprint(vs)
    # exit()
    vals = get_values(vs)
    # exit()
    # plot(vals); exit()
    vals = get_values(vs, 'eval', eps='0')
    do_eval_table(vals)
    print(0.05)
    vals = get_values(vs, 'eval', eps='0.05')
    do_eval_table(vals)