import glob
import pickle
from simulation.base.Agent import Agent
from typing import List
from simulation.main.Simulation import SimulationRunner, SingleAgentSimulation
from simulation.concrete.StateRepresentations import StateRepSectorsWithDistAndVelocity
import pyglet
from pyglet import shapes
import numpy as np
from simulation.main.Entity import Entity, EntityType
from agent.rl.simple_q_learning import SimpleQLearningAgent
from agent.erl.SimpleQERL import ERLAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.concrete.HardCodeAgentDistanceSectors import HardCodeAgentDistanceSectors
from simulation.concrete.RandomAgent import RandomAgent
from simulation.concrete.RandomAgent2 import RandomAgentDumb
class MyEntity:
    def __init__(self, entity: Entity, shapes, state_line):
        self.ent = entity
        self.shapes = shapes
        self.state_lines = state_line

    def update(self):
        for s in self.shapes:
            s.x = self.ent.pos[0]
            s.y = self.ent.pos[1]
        for s in self.state_lines:
            s.x = self.ent.pos[0]
            s.y = self.ent.pos[1]

def get_col(type):
    if type == EntityType.PREY: return (0,0,255)
    elif type == EntityType.FOOD: return (0,255,0)
steps = 0

def do_viz(simulation: SingleAgentSimulation, agent: Agent):
    runner = SimulationRunner(simulation, agent)
    init_time = pyglet.clock.time.time()
    size = simulation.size
    window = pyglet.window.Window(size, size)
    batch = pyglet.graphics.Batch()
    batch2 = pyglet.graphics.Batch()
    # x, y = size/2, size/2
    
    # create the things to draw
    
    # cs.append()
    def get_text():
        return f"Steps = {steps}.\nTime = {round(100 * (pyglet.clock.time.time() - init_time))/100}.\nFoods = {simulation.score[0]}.\nTotal Reward={simulation.total_rewards[0]}"
    document = pyglet.text.document.FormattedDocument(get_text())
    document.set_style(0,len(document.text),dict(color=(255,0,0,255), font_size=24))
    text = pyglet.text.layout.TextLayout(document,window.width, window.height//10 * 9.5,multiline=True, batch=batch2)
    """
    label = pyglet.text.Label(get_text(),
                            font_name='Times New Roman',
                            font_size=24,
                            x=window.width//6, y=window.height//10 * 9.5,
                            anchor_x='center', anchor_y='center', batch=batch2, color=(255,0,0,255))
    """
    all_my_ents: List[MyEntity] = []
    tmp = []
    for obs in simulation.obstacles:
        s = obs.start
        e = obs.end
        tmp.append(
            shapes.Line(s.x, s.y, e.x, e.y, color=(255, 255, 255), batch=batch)
        )
    for e in simulation.get_all_entities():
        shape = shapes.Circle(e.pos[0], e.pos[1], e.radius, color=get_col(e.type), batch=batch)
        s = [shape]
        lines = []
        if e.type == EntityType.PREY:
            s.append(
                shapes.Circle(e.pos[0], e.pos[1], simulation.VIZ_DIST, color=get_col(e.type), batch=batch)
            )
            s[-1].opacity=100

            for i in range(simulation.state_rep.NUM_CONES):
                # state food dist
                lines.append(
                    shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(0, 255, 255), batch=batch)
                )

                # dist wall
                lines.append(
                    shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(255, 0, 0), batch=batch, width=3)
                )

            #acceleration.
            s.append(
                shapes.Line(e.pos[0], e.pos[1], e.pos[0] + 30, e.pos[1] + 20, color=(255, 255, 255), batch=batch)
            )
            # s[-1].border=50
        all_my_ents.append(MyEntity(e, s, lines))
        pass

    def update(dt):
        global steps
        steps += 1
        runner.update()
        # label.text = get_text()
        document.text = get_text()
        for index in range(len(all_my_ents)-1, -1, -1):
            if all_my_ents[index].ent.is_dead:
                for s in all_my_ents[index].shapes:
                    s.delete()
                all_my_ents.pop(index)
        def normi(v):
            return v;
            return v / np.linalg.norm(v) * simulation.VIZ_DIST
        for index, e in enumerate(all_my_ents):
            if e.ent.type == EntityType.PREY:
                state = simulation._get_state(index)[0]
                state = state.get_value()
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
                for i in range(simulation.state_rep.NUM_CONES):
                    d_food = (state[2 + i * 2]  )*simulation.VIZ_DIST
                    d_wall = (state[2 + i * 2 + 1]) * simulation.VIZ_DIST
                    tmp1 = vs[i] * d_food
                    tmp2 = vs[i] * d_wall
                    # print(d_wall, np.linalg.norm(tmp2), np.linalg.norm(vs[i]), vs[i])
                    e.state_lines[i * 2].x2 = e.state_lines[i].x + tmp1[0]
                    e.state_lines[i * 2].y2 = e.state_lines[i].y + tmp1[1]

                    e.state_lines[i * 2 + 1].x2 = e.state_lines[i].x + tmp2[0]
                    e.state_lines[i * 2 + 1].y2 = e.state_lines[i].y + tmp2[1]

                # e.shapes[-3].x2 = state[0] + e.shapes[-3].x
                # e.shapes[-3].y2 = state[1] + e.shapes[-3].y
                # v = normi(e.ent.velocity)
                # e.shapes[-2].x2 = v[0] + e.shapes[-2].x
                # e.shapes[-2].y2 = v[1] + e.shapes[-2].y
                a = normi(e.ent.acc)
                e.shapes[-1].x2 = a[0] + e.shapes[-1].x
                e.shapes[-1].y2 = a[1] + e.shapes[-1].y
            e.update()

    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        batch2.draw()
        # pyglet.image.get_buffer_manager().get_color_buffer().save(name + '.png')
        # if not loop:
        # pyglet.app.exit()
    pyglet.clock.schedule_interval(update, 1/120.0)
    pyglet.app.run()
    pass

def q_learning():
    s = SingleAgentSimulation.basic_simul(num_obs=10, num_food=200)
    agent = SimpleQLearningAgent(s.state_rep.get_num_features(), len(SimulDiscreteAction))
    s = SingleAgentSimulation.combination_hard_coded_eval_level()

    do_viz(s, agent)
    pass

def main():
    s = SingleAgentSimulation.combination_hard_coded_eval_level()
    agent =  ERLAgent(s.state_rep.get_num_features(), len(SimulDiscreteAction))
    do_viz(s, agent)    


if __name__ == "__main__":
    np.random.seed(12345)
    q_learning()
    
