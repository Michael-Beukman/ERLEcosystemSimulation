from agent.multiagent.ERLMulti import ERLMulti
import datetime
from simulation.main.Entity import EntityType
# import pickle
import dill as pickle
import numpy as np
import os
from simulation.multi_agent_sim.hardcode_agents.HardCodeAgents import *
from simulation.multi_agent_sim.MultiStateRep import MultiAgentStateRep
# from simulation.multi_agent_sim.MultiAgentSimulation import MultiAgentSimulation
from simulation.discrete_ma_sim.discrete_multi_agent_sim import DiscreteMultiAgentSimulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')
# plt.switch_backend('Agg')
line1 = 1
maxes = -1


def plot_pickles(p='./pickles/experiment_multi_agent_test/05-16-2021_16-13-50_vals.p'):
    name = p.split("/")[-1].split(".p")[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim(bottom=0, top=40)

    with open(p, 'rb') as f:
        d = pickle.load(f)
    print(d.keys())
    l1 = plt.plot(d['food'][:1], label='food')[0]
    l2 = plt.plot(d['prey'][:1], label='prey')[0]
    l3 = plt.plot(d['pred'][:1], label='pred')[0]
    plt.legend()

    def animate(i):
        print(f"\r{i}/{len(d['food'])}", end='')
        a = np.arange(i)
        b, c, f = d['food'][:i],  d['prey'][:i], d['pred'][:i]
        l1.set_data(a, b)
        l2.set_data(a, c)
        l3.set_data(a, f)
        ax1.set_xlim(0, i + 1)
        # ax1.set_ylim(0, max([np.max(all_the_data[k]) for k in ['food', 'pred', 'prey']]) + 1)
        m = max([np.max(x) for x in [b, c, f]]) if i != 0 else 1
        ax1.set_ylim(0, m + 1)

        return l1, l2, l3
    ani = animation.FuncAnimation(
        fig, animate, frames=len(d['food']), interval=1, blit=False)
    ani.save(
        f"experiments/experiment_multi_agent/plots/{name}_{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.mp4", fps=10, writer='imagemagick')
    pass


def do_multi_agent_sim_stuff(plot_name, args):
    # keep track of the data
    all_the_data = {
        'food': [],
        'prey erl': [],
        'prey rando': [],
        'prey hard': [],
        'pred': [],

        'energy_pred': [],
        'energy_prey': [],


    }
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='erl', agent_prey='erl', n_preds=0, n_preys=40)
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='random', agent_prey='hardcode', n_preds=5, n_preys=30)
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(1024, agent_pred='random', agent_prey='hardcode', n_preds=5, n_preys=30, n_foods=300)
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(256, agent_pred='hard', agent_prey='hard', n_preds=20, n_preys=100, n_foods=500)
    sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(**args)
    steps = int(1e4)
    do_plot = True
    modder = 500
    # modder = 20
    preds = []
    preys = []
    food = []

    def save(data):
        d = f"./pickles/experiment_multi_agent_test"
        n = f'{d}/{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_vals.p'
        os.makedirs(d, exist_ok=True)
        with open(n, 'wb+') as f:
            pickle.dump({'data': data, 'sim': sim}, f)

    def animate(i):
        global maxes
        sim.update()
        if i % modder == 0 and i != 0:
            s = f"At Step = {i}. Prey = {len(sim.e_prey)}. Preds = {len(sim.e_predators)}. Food = {len(sim.foods)}. FPS = {round(1/sim.dt)}. DT = {round(sim.dt * 1000)}ms"
            print(s)

        def c(agents, clas):
            return sum([1 for i in agents.values() if isinstance(i, clas)])
        counts = {
            'hard': c(sim.a_prey, HardCodePrey),
            'rando': c(sim.a_prey, RandomMultiAgent),
            'erl': c(sim.a_prey, ERLMulti),
        }
        all_the_data['prey hard'].append(counts['hard'])
        all_the_data['prey rando'].append(counts['rando'])
        all_the_data['prey erl'].append(counts['erl'])

        all_the_data['pred'].append(len(sim.e_predators))
        all_the_data['food'].append(len(sim.foods) / 10)
        if do_plot:
            line1.set_data(np.arange(len(all_the_data['food'])), np.array(
                all_the_data['food']))
            # line2.set_data(np.arange(len(all_the_data['prey'])), np.array(all_the_data['prey']))

            line21.set_data(np.arange(len(all_the_data['prey hard'])), np.array(
                all_the_data['prey hard']))
            line22.set_data(np.arange(len(all_the_data['prey erl'])), np.array(
                all_the_data['prey erl']))
            line23.set_data(np.arange(len(all_the_data['prey rando'])), np.array(
                all_the_data['prey rando']))

            line3.set_data(np.arange(len(all_the_data['pred'])), np.array(
                all_the_data['pred']))
            maxes = max(maxes, max(len(sim.e_predators),
                        len(sim.e_prey), len(sim.foods) / 10))
            fig.canvas.draw()
        if len(sim.e_predators) == 0 and len(sim.e_prey) == 0 or i > 1e4:
            save(all_the_data)
            if do_plot:
                plt.savefig(
                    f"experiments/experiment_multi_agent/plots/test3_hard_{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.png")
                # ani.save(f"experiments/experiment_multi_agent/plots/test3_hard_gif_{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.mp4", fps=10, writer='imagemagick')
                ani.event_source.stop()
                # raise Exception
                exit()
            else:
                return False
        if do_plot:
            ax1.set_xlim(0, len(all_the_data['food']) + 1)
            # ax1.set_ylim(0, max([np.max(all_the_data[k]) for k in ['food', 'pred', 'prey']]) + 1)
            ax1.set_ylim(0, maxes + 1)
            return [line1, line21, line22, line23, line3]
        else:
            return True

    def gen():
        i = 0
        while len(sim.e_prey):
            yield i

    def init():
        global line1
        line1, = ax1.plot([0], [0], label='food')
        line2, = ax1.plot([0], [0], label='prey')
        line3, = ax1.plot([0], [0], label='pred')
        return [line1]
    if do_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylim(bottom=0, top=40)

        # global line1
        line1, = ax1.plot([0], [0], label='food')
        line21, = ax1.plot([0], [0], label='prey hard')
        line22, = ax1.plot([0], [0], label='prey erl')
        line23, = ax1.plot([0], [0], label='prey rando')
        line3, = ax1.plot([0], [0], label='pred')
        ax1.legend()
        # frames=gen,
        ani = animation.FuncAnimation(fig, animate, interval=1, blit=False)
        # plt.show()
        plt.show()
        # ani.save(f"experiments/experiment_multi_agent/plots/test3_hard_gif_{datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.mp4", fps=10, writer='imagemagick')
    else:
        for i in range(steps):
            if not animate(i):
                break

        plt.plot(np.arange(len(all_the_data['food'])), np.array(
            all_the_data['food']), label='food')
        plt.plot(np.arange(len(all_the_data['prey'])), np.array(
            all_the_data['prey']), label='prey')
        plt.plot(np.arange(len(all_the_data['pred'])), np.array(
            all_the_data['pred']), label='pred')
        plt.legend()
        plt.savefig(plot_name)
        plt.close()
        return i
    if do_plot:
        plt.show()


def grid_search():
    for npred in [0, 10, 50]:
        for nprey in [0, 10, 50, 100]:
            if npred == nprey == 0:
                continue
            for nfood in [50, 500]:
                name = f'experiments/experiment_multi_agent/plots/grid_search/{npred}_{nprey}_{nfood}.png'
                print("Doing {} now\n".format(name))
                steps = do_multi_agent_sim_stuff(name, dict(
                    agent_pred='erl', agent_prey='erl', n_preds=npred, n_preys=nprey, n_foods=nfood,
                    entity_lifetimes={
                        EntityType.FOOD: float('inf'),
                        # float('inf'), # 4000*100,
                        EntityType.PREY: 4000 * 0.5,
                        # float('inf'), #4000*100,
                        EntityType.PREDATOR: 4000 * 0.5
                    }
                ))
                print(
                    f"Pred = {npred}, Prey = {nprey}, Food = {nfood} got steps = {steps}")
    pass


def things():
    n_foods = 100
    d = dict(
        size=64,
        # n_preds=npred, n_preys=n_preys,
        n_foods=n_foods,
        n_preds={
            'erl': 0,
            'random': 0,
            'hard': 0
        }, n_preys={
            'erl': 5,
            'random': 5,
            'hard': 0
        },
        entity_lifetimes={
            EntityType.FOOD: float('inf'),
            # float('inf'), # 4000*100,
            EntityType.PREY: 200,
            # float('inf'), #4000*100,
            EntityType.PREDATOR: 4000 * 0.1
        },
        entity_energy_to_reproduce={
            EntityType.PREY: 1200,
            EntityType.PREDATOR: 2000
        },
    )
    do_multi_agent_sim_stuff(
        'experiments/experiment_multi_agent/plots/test_multi_species.png', d)


if __name__ == '__main__':
    # plot_pickles('./pickles/experiment_multi_agent_test/05-18-2021_08-37-04_vals.p')
    # do_multi_agent_sim_stuff()
    # grid_search()
    things()
