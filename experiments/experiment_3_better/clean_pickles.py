vals  = [
    './pickles/experiment_3/proper_v3/0/PPO/10000000.0/05-10-2021_15-30-02/models/final_save_rewards',
    './pickles/experiment_3/proper_v3/3/PPO/10000000.0/05-10-2021_15-34-08/models/final_save_rewards',
    './pickles/experiment_3/proper_v3/4/PPO/10000000.0/05-10-2021_15-34-10/models/final_save_rewards',
    './pickles/experiment_3/proper_v3/2/PPO/10000000.0/05-10-2021_15-33-52/models/final_save_rewards',
    './pickles/experiment_3/proper_v3/1/PPO/10000000.0/05-10-2021_15-32-28/models/final_save_rewards'
]
import pickle
for v in vals:
    dir = '/'.join(v.split("/")[:-1]) + "/final_save_rewards_small"
    print(dir)
    with open(v, 'rb') as f: 
        ans = pickle.load(f)
        del ans['all_rewards']
    with open(dir, 'wb+') as f:
        pickle.dump(ans, f)