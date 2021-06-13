import pickle
from simulation.base.Agent import Agent
from agent.evolutionary.EvolutionaryAgent import EvolutionaryAgent
from typing import List
import pyglet
from pyglet import shapes
import numpy as np
from simulation.main.Entity import Entity, EntityType
from agent.rl.simple_q_learning import SimpleQLearningAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.concrete.HardCodeAgentDistanceSectors import HardCodeAgentDistanceSectors
from simulation.concrete.RandomAgent import RandomAgent
from simulation.multi_agent_sim.MultiAgentSimulation import MultiAgentEntity, MultiAgentSimulation
from simulation.multi_agent_sim.MultiStateRep import MultiAgentStateRep
from timeit import default_timer as tmr
from matplotlib import pyplot as plt
import io
from pyglet import image
DO_RAD = False
r2 = np.sqrt(2)/2
vs = {
0: np.array([0, 1]),
2: np.array([1, 0]),
4: np.array([0, -1]),
6: np.array([-1, 0]),

1: np.array([r2, r2]),
3: np.array([r2, -r2]),
5: np.array([-r2, -r2]),
7: np.array([-r2, r2]),
}
class MyEntity:
    def __init__(self, entity: MultiAgentEntity, shapes, state_line):
        self.ent = entity
        self.shapes = shapes
        self.state_lines = state_line

    def update(self):
        for s in self.shapes:
            s.x = self.ent.pos[0]
            s.y = self.ent.pos[1]

def get_col(e):
    if e.type == EntityType.PREY: return (0,0,255)
    elif e.type == EntityType.FOOD: return (0,255,0)
    elif e.type == EntityType.PREDATOR: return (255, 0, 0)
steps = 0
food_count = 0
num_preys = 0
num_preds = 0
from time import time_ns
mytime = time_ns()

num_preds_list = list( num_preds * np.ones(100))
num_preys_list = list( num_preys * np.ones(100))
num_food_list = list( food_count * np.ones(100))

def do_viz(simulation: MultiAgentSimulation):
    global food_count, num_preys, mytime, num_preds, num_preds_list, num_preys, num_preys_list, food_count, num_food_list
    # runner = SimulationRunner(simulation, agent)
    plt.figure(figsize = (7,15), dpi=50)
    info_area_dimensions = { "x" : 500, "y" : simulation.SIZE}

    def get_graph_sprite() :
        #self.total_energy_in_system = 0
        #plt.figure(figsize = (3,3))
        global num_preds_list, num_preds, num_preys, num_preys_list, food_count, num_food_list
        num_preds_list += [num_preds]
        num_preds_list = num_preds_list[1:]
        num_preys_list += [num_preys]
        num_preys_list = num_preys_list[1:]
        num_food_list += [food_count]
        num_food_list = num_food_list[1:]
        plt.cla()
        plt.title("Entities over time")
        plt.xlabel(f"Time ({10} steps)")
        plt.ylabel("Number of entities")
        plt.plot(num_preds_list, label="Predator")
        plt.plot( num_preys_list, label ="Prey")
        plt.plot( num_food_list, label="Food")
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig( buffer, format='png')
        #plt.close()
        graph = image.load("file.png",file=buffer)
        #print( graph)
        #graph.blit( 100, 100, 100)
        #exit(1)
        sprite = pyglet.sprite.Sprite( graph, x = 0, y = 0)
        return sprite

    init_time = pyglet.clock.time.time()
    #size = simulation.size //TiZZAR
    window = pyglet.window.Window( simulation.SIZE+5, simulation.SIZE+5) #instead of passing in size
    # window2 = pyglet.window.Window( 350,simulation.SIZE)
    batch = pyglet.graphics.Batch()
    batch2 = pyglet.graphics.Batch()
    # x, y = size/2, size/2
    
    # create the things to draw
    # cs.append()
    def get_text(dt=None):
        global mytime
        ns = time_ns() - mytime
        secs = ns * 1e9
        mytime = time_ns()
        if dt is not None:
            return f"Prey = {len(simulation.e_prey)}. Preds = {len(simulation.e_predators)}. Food = {len(simulation.foods)}. FPS = {round(1/dt)}. DT = {round(dt * 1000)}ms. Steps= {steps}"    
        return f"FPS = {1/secs}"
    document = pyglet.text.document.FormattedDocument(get_text())
    document.set_style(0,len(document.text),dict(color=(255,0,0,255), font_size=24))
    text = pyglet.text.layout.TextLayout(document,window.width, window.height//10 * 9.5,multiline=True, batch=batch2)
    all_my_ents: List[MyEntity] = []
    tmp = []

    def get_shapes_lines_for_entity(e):
        s = []
        lines = []
        if DO_RAD:
            s.append(
                    shapes.Circle(e.pos[0], e.pos[1], simulation.VIZ_DIST, color=get_col(e), batch=batch)
                )
            s[-1].opacity = 100
            s.append(
                    shapes.Arc( e.pos[0], e.pos[1], min( simulation.VIZ_DIST, simulation.VIZ_DIST * e.energy/simulation.entity_energy_to_reproduce[e.type]), color=(255,255,255), batch=batch)
            )
            s[-1].opacity = 200
        #s[-1].opacity= (300 - 20) * ( e.energy/e.MAX_ENERGY) + 20
        #given MAX_ENERGY an entity can have
        #we map opacity in range(30,200) ~ chosen subjectively

        for i in range(simulation.state_rep.NUM_CONES):
            if not DO_RAD: break
            # state food dist
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(0, 255, 0), batch=batch, width=1)
            )

            # state prey
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(255, 255, 255), batch=batch, width=1)
            )

            # state pred
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(120, 255, 0), batch=batch, width=1)
            )

            # dist wall
            lines.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(255, 0, 0), batch=batch, width=1)
            )

        # #acceleration.
        # s.append(
        #     shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(255, 255, 255), batch=batch)
        # )
        return s, lines

    for obs in simulation.obstacles:
        s = obs.start
        e = obs.end
        tmp.append(
            shapes.Line(s.x, s.y, e.x, e.y, color=(255, 255, 255), batch=batch)
        )
    for e in simulation.get_all_entities():
        shape = shapes.Circle(e.pos[0], e.pos[1], e.radius, color=get_col(e), batch=batch)
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

    def update(dt):
        su = tmr()
        # print(dt)
        global steps, food_count, num_preys, num_preds
        steps += 1
        ssu = tmr()
        simulation.update()
        esu = tmr()
        # print(f"Time for sim update takes: {round((esu-ssu)*1000)}ms")
        # label.text = get_text()
        document.text = get_text(dt)
        for index in range(len(all_my_ents)-1, -1, -1):
            if all_my_ents[index].ent.is_dead:
                if all_my_ents[index].ent.type == EntityType.FOOD: food_count -= 1
                for s in all_my_ents[index].shapes:
                    s.delete()
                for s in all_my_ents[index].state_lines:
                    s.delete()
                all_my_ents.pop(index)
        for index in range(len(simulation.foods) - food_count):
            e = simulation.foods[index + food_count]
            shape = shapes.Circle(e.pos[0], e.pos[1], e.radius, color=get_col(e), batch=batch)
            all_my_ents.append(MyEntity(e, [shape], []))
        food_count = len(simulation.foods)

        for index in range(len(simulation.a_prey) - num_preys):
            e = simulation.e_prey[index + num_preys]
            shape = shapes.Circle(e.pos[0], e.pos[1], e.radius, color=get_col(e), batch=batch)
            s, l = get_shapes_lines_for_entity(e)
            all_my_ents.append(MyEntity(e, [shape] + s, l))
        
        for index in range(len(simulation.a_predators) - num_preds):
            e = simulation.e_predators[index + num_preds]
            shape = shapes.Circle(e.pos[0], e.pos[1], e.radius, color=get_col(e), batch=batch)
            s, l = get_shapes_lines_for_entity(e)
            all_my_ents.append(MyEntity(e, [shape] + s, l))
            
        num_preys = len(simulation.a_prey)
        num_preds = len(simulation.a_predators)

        def normi(v):
            return v;
        sl = tmr()
        # print(f"Time for beginning {round(sl - esu) * 1000} ms")
        t = 0
        kk = tmr()
        for index, e in enumerate(all_my_ents):
            if e.ent.type == EntityType.PREY or e.ent.type == EntityType.PREDATOR:
                e.update()
        kkk = tmr()
        t+=kkk-kk
        eu = tmr()
        # print(f"Time for viz update takes: {round((eu-su)*1000)}ms")
        # print(f"Time for viz final loop takes: {round((eu - sl)*1000)}ms. cone loop {round(t * 1000)}ms")
    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        batch2.draw()

    mytime = time_ns()
    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()
    pass

def test():
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent='erl')
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='erl', agent_prey='erl', n_preds=20, n_preys=30)
    # sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='erl', agent_prey='erl', n_preds=10, n_preys=40)
    sim = MultiAgentSimulation.get_proper_sim_for_learning(MultiAgentStateRep(), agent_pred='random', agent_prey='hardcode', n_preds=5, n_preys=30)
    do_viz(sim)
if __name__ == "__main__":
    np.random.seed(12345)
    test()
