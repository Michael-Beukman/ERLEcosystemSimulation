import glob
from pprint import pprint
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def get_diff_hp_pickle_names():
    vals = defaultdict(lambda : {})
    for i in range(5):
        # alpha_0.0005__gamma_0.99__lambd_0.95__eps_init_0.1__eps_min_0.02__eps_decay_0.999
        s1 = f'pickles/experiment_5/proper_v5/{i}/ERL/10000000.0/50/alpha_0.05__gamma_1__lambd_0__eps_init_0.1__eps_min_0.02__eps_decay_0.999/*/metrics'
        # s = f'pickles/experiment_6/proper_v6/{i}/ERL/10000000.0/50/alpha_0.05__gamma_1__lambd_0__eps_init_0.1__eps_min_0.02__eps_decay_0.999/*/metrics'
        s = f'pickles/experiment_6/proper_v6/{i}/ERL/10000000.0/50/alpha_0.05__gamma_1__lambd_0__eps_init_0.1__eps_min_0.02__eps_decay_0.999/05-18-2021_04*/metrics'
        # s = f'pickles/experiment_6/proper_v6/{i}/ERL/10000000.0/50/alpha_0.05__gamma_1__lambd_0__eps_init_0.1__eps_min_0.02__eps_decay_0.999/05-18-2021_07*/metrics'
        # pickles/experiment_6/proper_v6/0/ERL/10000000.0/50/alpha_0.05__gamma_1__lambd_0__eps_init_0.1__eps_min_0.02__eps_decay_0.999/05-18-2021_07-07-44/metrics
        print("Glob.glob ", len(glob.glob(s)))
        for g in glob.glob(s):
            key = g.split('/')[-3]
            key = key.split("__eps_")[0]
            key = ', '.join(map(lambda s: s.title(), key.split("__")))
            key = key.replace("_", " = ") + " with bias"
            key = "ERL with bias"
            vals[i][key] = g
        
        for g in glob.glob(s1):
            key = g.split('/')[-3]
            key = key.split("__eps_")[0]
            key = ', '.join(map(lambda s: s.title(), key.split("__")))
            key = key.replace("_", " = ") + " without bias"
            key = "ERL without bias"
            vals[i][key] = g
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
    for key, li in vals.items():
        li = np.array(li)
        print(li.shape, key,li)

        # if 'Gamma = 0.99' in key: continue
        size = 10
        arr = li
        arr2 = np.zeros_like(li)
        print(arr2.shape, 's')
        for seed in range(arr.shape[0]):
            for ep in range(arr.shape[1]):
                
                arr2[seed, ep] = np.mean(arr[seed, max(0, ep-size): ep + 1])
        li = arr2
        mean = np.mean(li, axis=0)
        std = np.std(li, axis=0)
        name = f"ERL size = {key}"
        ins = np.linspace(0, 1e7, len(mean))
        plt.plot(ins, mean, label=name)
        plt.fill_between(ins, mean - std, mean + std, alpha=0.5)
    plt.ylim(-100, 100)
    plt.legend()
    plt.xlabel("Timesteps trained")
    plt.ylabel("Average Reward while training")
    # plt.show()
    plt.title("Comparing using a bias vs not")
    plt.savefig("experiments/experiment_6/results/train.png")

def plot_v2(vals):
    # alphas
    ins = [0.0005, 0.005, 0.05]
    lines = defaultdict(lambda: {})
    for key, li in vals.items():
        li = np.array(li)
        print(li.shape, key)

        mean = np.mean(li, axis=0)
        point = np.mean(mean[-10:])
        a, b, c = key.split(",")
        key = ''.join([b, c])
        lines[key][float(a.split(" = ")[1])] = point
    print(lines)
    for key, val in lines.items():
        outs = [
            val[i] for i in ins
        ]
        plt.plot(ins, outs, label=key)
    plt.legend()
    plt.xscale('log')
    plt.show()
    pass

def do_eval_table(vals, suffix):
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
                    N = -2000
                    idx = li < N
                    # print("NUm bad ", idx.sum(), (li < -10000).sum())
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
            k: f"${np.round(np.mean(sorted(kk)[1:-1]))} \pm {np.round(np.std(sorted(kk)[1:-1]))}$" for k,kk in values.items()
        }
    # print(all_vals)
    df = pd.DataFrame(all_vals)
    print('cols', df.columns)
    df = df.T.sort_values(by='fixed_random')
    
    df['Total'] = df['fixed_random'].map(lambda x: float(x.split(" ")[0][1:])) 
    df['Total'] += df['obstacles'].map(lambda x: float(x.split(" ")[0][1:]))
    df['Total'] += df['foodhunt'].map(lambda x: float(x.split(" ")[0][1:]))
    df['Total'] += df['combination'].map(lambda x: float(x.split(" ")[0][1:]))
    df = df.sort_values(by='Total')
    def temp(s):
        return "$" + s.replace("Alpha", r"\alpha") \
                .replace("Lambd", r"\lambda") \
                .replace("Gamma", r"\gamma") + "$"
    # df = df.drop(["Tmp"], axis=1)
    df = df.reset_index()
    df.columns = ['Parameters'] + list(df.columns[1:])
    df['Parameters'] = df['Parameters'].map(temp)
    print('abc', df.columns)
    print(df.index); 
    print(df.to_latex(f"experiments/experiment_6/results/eval_tables_{suffix}.tex", index=False, escape=False))
    print(df); #exit()



if __name__ == "__main__":
    vs = get_diff_hp_pickle_names()
    pprint(vs)
    # exit()
    vals = get_values(vs)
    # exit()
    # plot(vals);  # exit()
    # plot_v2(vals);   exit()
    vals = get_values(vs, 'eval', eps='0')
    do_eval_table(vals, suffix="eps_0")
    vals = get_values(vs, 'train', eps='0')
    plot(vals)
    exit()
    print(0.05)
    vals = get_values(vs, 'eval', eps='0.05')
    do_eval_table(vals, suffix="eps_0.05")