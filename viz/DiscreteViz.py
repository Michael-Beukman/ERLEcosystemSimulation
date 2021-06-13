from time import time_ns
from timeit import default_timer as tmr
from typing import List

import numpy as np
import pyglet
from agent.multiagent.ERLMulti import ERLMulti
from pyglet import shapes
from simulation.discrete_ma_sim.discrete_multi_agent_sim import (
    DiscMultiAgentEntity, DiscreteMultiAgentSimulation)
from simulation.main.Entity import EntityType
from simulation.multi_agent_sim.hardcode_agents.HardCodeAgents import *

DO_RAD = False
DO_STATE_LINES = False
r2 = np.sqrt(2)/2
vs = {
    6: np.array([0, 1])[::-1],
    4: np.array([1, 0])[::-1],
    1: np.array([0, -1])[::-1],
    3: np.array([-1, 0])[::-1],

    7: np.array([r2, r2])[::-1],
    2: np.array([r2, -r2])[::-1],
    0: np.array([-r2, -r2])[::-1],
    5: np.array([-r2, r2])[::-1],
}
previous_current_id = 0


class MyEntity:
    def __init__(self, entity: DiscMultiAgentEntity, shapes, state_line):
        self.ent = entity
        self.shapes = shapes
        self.state_lines = state_line

    def update(self):
        for s in self.shapes:
            # (e.pos[0] + 0.5) * TILE_WIDTH, (e.pos[1] + 0.5) * TILE_HEIGHT
            s.x = (self.ent.pos[1] + 0.5) * TILE_WIDTH
            s.y = window.height - (self.ent.pos[0] + 0.5) * TILE_HEIGHT
        for s in self.state_lines:
            # break
            s.x = (self.ent.pos[1] + 0.5) * TILE_WIDTH
            s.y = window.height - (self.ent.pos[0] + 0.5) * TILE_HEIGHT


def get_col(e, which_agent=None):
    if which_agent is not None:
        if e.type == EntityType.PREY:
            if isinstance(which_agent, HardCodePrey):
                return (255, 0, 255)
            elif isinstance(which_agent, RandomMultiAgent):
                return (255, 255, 0)
            else:
                # erl
                return (0, 255, 255)
        else:
            if isinstance(which_agent, HardCodePred):
                return (255, 0, 0)
            elif isinstance(which_agent, RandomMultiAgent):
                return (255, 120, 0)
            else:
                # erl
                return (120, 120, 120)


    # if e.type == EntityType.PREY: return (0,0,255)
    if e.type == EntityType.PREY:
        return (0, 255, 255)
    elif e.type == EntityType.FOOD:
        return (0, 255, 0)
    elif e.type == EntityType.PREDATOR:
        return (255, 0, 0)


steps = 0
food_count = 0
num_preys = 0
num_preds = 0
mytime = time_ns()

num_preds_list = list(num_preds * np.ones(100))
num_preys_list = list(num_preys * np.ones(100))
num_food_list = list(food_count * np.ones(100))

TILE_WIDTH = 16 * 4
TILE_HEIGHT = 16 * 4
_dt = 1
window = None
DO_DUMMY = False
CIRCLE_SIZE = None


def do_viz(simulation: DiscreteMultiAgentSimulation, grid=True):
    # SCREEN_SIZE = 1024
    global CIRCLE_SIZE
    SCREEN_SIZE = simulation.size * TILE_WIDTH
    global food_count, num_preys, mytime, num_preds, num_preds_list, num_preys, num_preys_list, food_count, num_food_list, window, previous_current_id
    # runner = SimulationRunner(simulation, agent)

    init_time = pyglet.clock.time.time()
    #size = simulation.size //TiZZAR
    # instead of passing in size
    window = pyglet.window.Window(SCREEN_SIZE, SCREEN_SIZE)
    # window2 = pyglet.window.Window( 350,simulation.SIZE)
    batch = pyglet.graphics.Batch()
    batch2 = pyglet.graphics.Batch()
    # x, y = size/2, size/2

    # create the things to draw
    # cs.append()
    def get_text(dt=None):
        # return ""
        global mytime
        ns = time_ns() - mytime
        secs = ns * 1e9
        mytime = time_ns()
        if dt is not None:
            def c(agents, clas):
                return sum([1 for i in agents.values() if isinstance(i, clas)])
            counts = {
                'hard': c(simulation.a_prey, HardCodePrey),
                'rando': c(simulation.a_prey, RandomMultiAgent),
                'erl': c(simulation.a_prey, ERLMulti),
            }

            counts2 = {
                'hard': c(simulation.a_predators, HardCodePred),
                'rando': c(simulation.a_predators, RandomMultiAgent),
                'erl': c(simulation.a_predators, ERLMulti),
            }
            return f"Hard = {counts['hard']}. Rando={counts['rando']}. ERL={counts['erl']}. Food = {len(simulation.foods)}. FPS = {round(1/dt)}. DT = {round(dt * 1000)}ms. Steps= {steps}\nPredHard = {counts2['hard']}. PredRando={counts2['rando']}. PredERL={counts2['erl']}."
            # return f"Prey = {len(simulation.e_prey)}. Preds = {len(simulation.e_predators)}. Food = {len(simulation.foods)}. FPS = {round(1/dt)}. DT = {round(dt * 1000)}ms. Steps= {steps}"
        return f"FPS = {1/secs}"
    test = []
    if grid:
        for i in range(simulation.size):
            x = i * TILE_WIDTH
            test.append(shapes.Line(x, 0, x, simulation.size *
                        TILE_HEIGHT, color=(255, 255, 255), batch=batch))
        for j in range(simulation.size):
            y = j * TILE_HEIGHT
            test.append(shapes.Line(0, y, simulation.size *
                        TILE_WIDTH, y, color=(255, 255, 255), batch=batch))

    document = pyglet.text.document.FormattedDocument(get_text())
    document.set_style(0, len(document.text), dict(
        color=(255, 255, 255, 255), font_size=24))
    text = pyglet.text.layout.TextLayout(
        document, window.width, window.height//10 * 9.9, multiline=True, batch=batch2)
    all_my_ents: List[MyEntity] = []
    tmp = []

    def get_shapes_lines_for_entity(e):
        s = []
        lines = []
        # return s, lines

        for i in range(8):
            if not DO_STATE_LINES:
                break
            # state food dist
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] +
                            20, color=(0, 255, 0), batch=batch, width=1)
            )

            # state prey
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] +
                            20, color=(255, 255, 255), batch=batch, width=1)
            )

            # state pred
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] +
                            20, color=(120, 255, 0), batch=batch, width=1)
            )

            # dist wall
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] +
                            20, color=(255, 0, 0), batch=batch, width=1)
            )
        return s, lines
    # print(simulation.get_all_entities())
    # print(simulation.smart_agents())
    for id, e in simulation.get_all_entities().items():
        shape = shapes.Circle((e.pos[1] + 0.5) * TILE_WIDTH, window.height - (
            # e.pos[0] + 0.5) * TILE_HEIGHT, CIRCLE_SIZE, color=get_col(e, None if e.type != EntityType.PREY else simulation.smart_agents[id]), batch=batch)
            e.pos[0] + 0.5) * TILE_HEIGHT, CIRCLE_SIZE, color=get_col(e, None if e.type == EntityType.FOOD else simulation.smart_agents[id]), batch=batch)
        s = [shape]
        lines = []
        if e.type == EntityType.PREY or e.type == EntityType.PREDATOR:
            if e.type == EntityType.PREY:
                num_preys += 1
            else:
                num_preds += 1
            ss, lines = get_shapes_lines_for_entity(e)
            s.extend(ss)
        else:
            food_count += 1
        all_my_ents.append(MyEntity(e, s, lines))
        pass
    previous_current_id = simulation.current_id

    ree = []
    # ree.append(shapes.Circle(0, 0, 50, color=(255, 255, 255), batch=batch))
    print("all entites", all_my_ents)

    @window.event
    def on_key_press(symbol, modifiers):
        if not DO_DUMMY:
            return
        print("Presed")
        update()
        pass

    def dummy_update(dt):
        global _dt
        _dt = dt

    def update(dt=None):
        # return
        global previous_current_id
        if dt is None:
            dt = _dt
        # print("Update")
        su = tmr()
        # print(dt)
        global steps, food_count, num_preys, num_preds
        steps += 1
        ssu = tmr()
        simulation.update()
        esu = tmr()
        print(f"Time for sim update takes: {round((esu-ssu)*1000)}ms")
        # label.text = get_text()
        document.text = get_text(dt)
        for index in range(len(all_my_ents)-1, -1, -1):
            if all_my_ents[index].ent.is_dead:
                if all_my_ents[index].ent.type == EntityType.FOOD:
                    food_count -= 1
                for s in all_my_ents[index].shapes:
                    s.delete()
                for s in all_my_ents[index].state_lines:
                    s.delete()
                all_my_ents.pop(index)
        if 1:
            # Add new entities
            alls = simulation.get_all_entities()
            for i in range(previous_current_id, simulation.current_id):
                # print("ADDING WITH ID = ", i)
                e = alls[i]
                shape = shapes.Circle((e.pos[1] + 0.5) * TILE_WIDTH, window.height - (
                    e.pos[0] + 0.5) * TILE_HEIGHT, CIRCLE_SIZE, color=get_col(e,
                                                                            #   None if e.type != EntityType.PREY else simulation.smart_agents[i]
                                                                              None if e.type == EntityType.FOOD else  simulation.smart_agents[i]
                                                                              ), batch=batch)
                all_my_ents.append(MyEntity(e, [shape], []))
            previous_current_id = simulation.current_id
        num_preys = len(simulation.a_prey)
        num_preds = len(simulation.a_predators)

        def normi(v):
            return v
            return v / np.linalg.norm(v) * simulation.VIZ_DIST
        sl = tmr()
        # print(f"Time for beginning {round(sl - esu) * 1000} ms")
        t = 0
        kk = tmr()
        for index, e in enumerate(all_my_ents):
            if e.ent.type == EntityType.PREY or e.ent.type == EntityType.PREDATOR:
                if DO_STATE_LINES:
                    # if index == 0:print(e.ent.pos)
                    index = DiscreteMultiAgentSimulation.IPREY if e.ent.type == EntityType.PREY else DiscreteMultiAgentSimulation.IPRED
                    id = simulation.vals(index, e.ent.pos[0], e.ent.pos[1])
                    # print(simulation.states, id, e.ent.pos)
                    # print(simulation.values[index])
                    state = simulation.states[id]
                    # print(e.ent.type, state)
                    for i in range(8):
                        d_food = (state[2 + i * 4])  # * simulation.VIZ_DIST
                        d_prey = (state[3 + i * 4])  # * simulation.VIZ_DIST
                        d_pred = (state[4 + i * 4])  # * simulation.VIZ_DIST
                        d_wall = (state[5 + i * 4])  # * simulation.VIZ_DIST
                        four_i = i * 4
                        for index, val in enumerate([d_food, d_prey, d_pred, d_wall]):
                            tmp1 = vs[i] * val
                            j = four_i + index
                            added = e.ent.pos + tmp1
                            if len(e.state_lines) <= j:
                                continue
                            e.state_lines[j].position = (e.ent.pos[0], e.ent.pos[1], (
                                added[1] + 0.5) * TILE_WIDTH, window.height - (added[0]+0.5) * TILE_HEIGHT)
                e.update()
        kkk = tmr()
        t += kkk-kk
        eu = tmr()
        # print(f"Time for viz update takes: {round((eu-su)*1000)}ms")
        # print(f"Time for viz final loop takes: {round((eu - sl)*1000)}ms. cone loop {round(t * 1000)}ms")

    @window.event
    def on_draw():
        # pyglet.gl.glClearColor(1,192/255,203/255,1)
        window.clear()
        batch.draw()
        batch2.draw()
        # get_graph_sprite().draw()
        # pyglet.image.get_buffer_manager().get_color_buffer().save(name + '.png')
        # if not loop:
        # pyglet.app.exit()
    # @window2.event

    def on_draw():
        return
        if(steps % 10 == 0):
            window2.clear()
            get_graph_sprite().draw()

    mytime = time_ns()
    if not DO_DUMMY:
        pyglet.clock.schedule_interval(update, 1/60.0)
    else:
        pyglet.clock.schedule_interval(dummy_update, 1/60.0)
    pyglet.app.run()
    pass


def test():
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent='erl')
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='erl', agent_prey='erl', n_preds=20, n_preys=30)
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='erl', agent_prey='erl', n_preds=10, n_preys=40)
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(256, agent_pred='random', agent_prey='random', n_preds=5, n_preys=30)
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(16, agent_pred='random', agent_prey='random', n_preds=50, n_preys=50, n_foods=1000)
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(16, agent_pred='random', agent_prey='random', n_preds=1, n_preys=1, n_foods=10)
    global TILE_HEIGHT, TILE_WIDTH
    TILE_WIDTH = TILE_HEIGHT = 16
    global CIRCLE_SIZE, DO_DUMMY
    CIRCLE_SIZE = TILE_HEIGHT/2
    # DO_DUMMY = 1
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(128, agent_pred='hard', agent_prey='hard', n_preds=50, n_preys=50, n_foods=50)
    # sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(64, agent_pred='hard', agent_prey='hard', n_preds=20, n_preys=50, n_foods=500)
    npred = 5
    n_preys = 40
    n_foods = 100
    sim = DiscreteMultiAgentSimulation.get_proper_sim_for_learning(size=64, **dict(
        # n_preds=npred, n_preys=n_preys,
        n_foods=n_foods,
        n_preds={
            'erl': 10,
            'random': 10,
            'hard': 0
        }, n_preys={
            'erl': 10,
            'random': 10,
            'hard': 10
        },
        entity_lifetimes={
            EntityType.FOOD: float('inf'),
            # float('inf'), # 4000*100,
            EntityType.PREY: 300,
            # float('inf'), #4000*100,
            EntityType.PREDATOR: 600
        },
        entity_energy_to_reproduce={
            EntityType.PREY: 1200,
            EntityType.PREDATOR: 1200
        },
    ))
    print(sim.values[0])
    print(sim.values[1])
    print(sim.values[2])
    print(sim.all_entities())
    do_viz(sim, grid=False)


if __name__ == "__main__":
    np.random.seed(12345 + 2)
    test()
