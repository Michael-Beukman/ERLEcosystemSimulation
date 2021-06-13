from collections import defaultdict
from email.policy import default
import glob
import os
import pickle
from typing import Dict, List
from matplotlib import pyplot as plt
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
# 682 397
import numpy as np
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def test_vostro_preds():
    names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/v4_vostro/different_preds/**/*.p', recursive=True)
    dic = defaultdict(lambda : {})
    for n in names:
        # print(n)
        # pickles/experiment_multi_agent_test/proper_exps/v4_vostro/different_preds/erl_5_rando_5_hard_0/5/05-30-2021_16-32-11/data.p
        seed = n.split('/')[-3]
        # print(seed)
        with open(n, 'rb') as f:
            alls = pickle.load(f)
            j = alls['data']
            # print("hard =", j['pred hard'][0])
            # print("erl =", j['pred erl'][0])
            # print("rando =", j['pred rando'][0])
            # print("--")
            description = 'pred_hard_{}_erl_{}_rando_{}'.format(j['pred hard'][0], j['pred erl'][0], j['pred rando'][0])
            dic[description][seed] = alls
            print(description, seed)

    for desc, dic_of_seeds in dic.items():
        print(desc)

    print(len(names))

def make_pretty(name: str):
    name = name.replace('rando', 'random')
    splits = name.split(" ")
    if len(splits) == 1: return name.title()
    return f'{splits[0].title()} ({splits[1].title()})'

def filter_out_outliers(alls: Dict[str, List[List[int]]]):
    counts_end = {
        k: [] for k in alls
    }
    for key in alls:
        print("LL", key)
        for list in alls[key]:
            counts_end[key].append(list[-1])
    print(counts_end)
    KEY_TO_CONSIDER = 'prey erl'
    if KEY_TO_CONSIDER not in alls:
        KEY_TO_CONSIDER = 'prey rando'
    thingy = [(value, index) for index, value in enumerate(counts_end[KEY_TO_CONSIDER])]
    print()
    values = sorted(thingy)[1: -1]
    to_return = defaultdict(lambda : [])
    indices_to_keep = [index for _, index in values]

    for key in alls:
        for index, list in enumerate(alls[key]):
            if index in indices_to_keep:
                to_return[key].append(list)
    return to_return;
    exit()

def analyse_from_vostro():
    TO_SAVE = 'simple_test_preds'
    TO_SAVE = 'preys'
    TO_SAVE = 'preds_and_preys_0603'
    # names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/v4_vostro/different_preys/*')
    names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/v6_proper_simple/preys_pred/*')
    # names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/v4_vostro/different_preds/*')
    # names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/v5_simple/just_preys/*')
    print(names)
    for folder in names:
        dic = {

        }
        desc = folder.split("/")[-1]
        if 'Icon' in desc: continue
        alls = defaultdict(lambda : [])
        for seed in range(5):
            fff =  os.path.join(folder, str(seed), '*', 'data.p')
            if 'Icon' in fff: continue
            files = glob.glob(fff)
            assert len(files) == 1, f"bad {len(files)}, {fff}"
            with open(files[0], 'rb') as f:
                obj = pickle.load(f)['data']
            dic[seed] = obj
            for key in obj.keys():
                alls[key].append(obj[key])


        
        # print("Len", len(alls[key]), len(alls[key][0]), len(alls[key][1]))
        for key in alls:
            print(key)
            l = np.array(alls[key]); print(l.shape)
            alls[key] = l
        alls = {k: v for k, v in alls.items() if v.shape[1] != 0}
        new_alls = {}

        alls = {k: v for k, v in alls.items() if v[0][0] != 0}
        try:
            alls = filter_out_outliers(alls)
        except Exception as e:
            continue;

        for key, nparray in alls.items():
            new_alls[key] = np.mean(nparray, axis=0)
            std = np.std(nparray, axis=0)
            if new_alls[key][0] == 0 or key == 'food': continue
            # new_alls[key] = np.clip(new_alls[key], 0, 100)
            plt.plot(new_alls[key], label=make_pretty(key))
            plt.fill_between(np.arange(len(new_alls[key])), new_alls[key] - std, new_alls[key] + std, alpha=0.5)
        # plt.ylim(0, 100)
        plt.legend()
        plt.title("Number of entities over time")
        plt.xlabel("Timesteps")
        plt.ylabel("Number of Entities")
        plt.savefig(f'experiments/experiment_multi_agent/plots/for_report/{TO_SAVE}_{desc}.png')
        # plt.show()
        plt.close()

        print(list(alls.keys()))
        # return

    pass

def main():
    # names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/different_preys/*/*/data.p')
    names = glob.glob(f'pickles/experiment_multi_agent_test/proper_exps/v3/different_preys/*/*/data.p')
    all_names = defaultdict(lambda : [])
    for n in names:
        with open(n, 'rb') as f:
            j = pickle.load(f)
            a, b = j['data'], j['sim']
            print("Size data = ", total_size(a))
            print("Size SIm = ", total_size(b))
            print("Size total = ", total_size(j))
            for key, list in j['data'].items():
                all_names[key].append(list)
    
    for n, list in all_names.items():
        val = np.array(list)
        print(n, val.shape)
        if val.shape != (5, 10000):
            print("BAD")
            continue
        if n == 'food':
            val = val / 10.0
        val = sorted(val, key=lambda li: li[-1])[1:]
        mean = np.mean(val, axis=0)
        std = np.std(val, axis=0)

        plt.plot(mean, label=n)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.5)
    plt.legend()
    plt.show()
# main()
analyse_from_vostro()
# test_vostro_preds()
