from agent.erl.SimpleQERL import ERLAgent
import gym
from agent.rl.simple_q_learning import SimpleQLearningAgent
from simulation.base.Action import Action
from typing import List, Tuple
from simulation.main.Entity import Entity, EntityType
from simulation.base.Agent import Agent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.concrete.RandomAgent import RandomAgent
from simulation.concrete.HardCodeAgentDistanceSectors import HardCodeAgentDistanceSectors
import simulation.concrete.StateRepresentations as StateRep
from agent.erl.SimpleQERL import ERLAgent
import numpy as np
import simulation.utils.utils as sg
class SingleAgentSimulation:
    OOB_REWARD = -100 # out of bounds
    SINGLE_STEP_REWARD = 0
    VIZ_DIST = 75
    UPDATE_EVERY = 1
    SIZE = 1024
    def __init__(self, smart_ents: List[Entity], stat_ents: List[Entity], 
                    state_rep: "StateRep.StateRepresentation", extra_obstacles: List[sg.Segment2]=None) -> None:
        assert len(smart_ents) == 1
        self.size = SingleAgentSimulation.SIZE
        self.smart_entities: List[Entity] = smart_ents
        self.stationary_entities: List[Entity] = stat_ents
        # how much food you collected.
        self.score = np.zeros_like(smart_ents)
        self.total_rewards = np.zeros_like(smart_ents)
        r2 = np.sqrt(2)/2
        self.velocities = {
            SimulDiscreteAction.UP: np.array([0, 1.0]),
            SimulDiscreteAction.DOWN: np.array([0, -1.0]),
            SimulDiscreteAction.LEFT: np.array([-1.0, 0]),
            SimulDiscreteAction.RIGHT: np.array([1.0, 0]),

            SimulDiscreteAction.UP_RIGHT: np.array([r2, r2]),
            SimulDiscreteAction.DOWN_RIGHT: np.array([r2, -r2]),
            SimulDiscreteAction.DOWN_LEFT: np.array([-r2, -r2]),
            SimulDiscreteAction.UP_LEFT: np.array([-r2, r2]),
        }

        self.rewards = {
            EntityType.FOOD: 1
        }

        self.dt = 0.1
        self.state_rep = state_rep

        self.obstacles: List[sg.Segment2] = self.get_obstacles(extra_obstacles)
        self.extra_obstacles = extra_obstacles if extra_obstacles is not None else []

    def get_obstacles(self, extra_obstacles) -> List[sg.Segment2]:
        "This returns a list of line segments representing the obstacles like walls and rectangles."
        tl = sg.Point2(0,0)
        bl = sg.Point2(0,self.size)
        tr = sg.Point2(self.size,0)
        br = sg.Point2(self.size,self.size)

        ans = [
            sg.Segment2(br, bl),
            sg.Segment2(br, tr),
            sg.Segment2(tl, tr),
            sg.Segment2(tl, bl),
        ]
        if extra_obstacles is not None:
            ans.extend(extra_obstacles)
        
        return ans

    def get_all_entities(self) -> List[Entity]:
        """ This returns a list of all of the current entities. You can use Entity.type to determine which are which. """
        return self.smart_entities + self.stationary_entities
    
    def update(self, agent_action: SimulDiscreteAction, dist_indices: np.array):
        "This performs one step of the simulation and returns a single reward value."
        rewards = 0
        i = 0
        # for i, action in enumerate(actions):
        rewards += self._perform_action(self.smart_entities[i], agent_action, dist_indices)
        
        return rewards
        
    
    def get_num_features(self):
        return self.state_rep.get_num_features()

    def _get_state(self, i: int) -> Tuple[State, np.array]:
        return self.state_rep.get_state(i, self)

    def _perform_action(self, e: Entity, a: SimulDiscreteAction, indices_of_close_points: np.array) -> float:
        "This takes in the action and entity and actually performs that action, and simulates the consequences. It returns the reward."

        indices_of_close_points = indices_of_close_points[:len(self.stationary_entities)]
        index_of_e = self.smart_entities.index(e)
        acc = self.velocities[a] * 5
        old_pos = e.pos.copy()
        e.update(acc, self.dt)
        new_pos = e.pos
        if (self._outside(new_pos, e.radius)):
            e.pos = old_pos
            e.velocity *= 0
            self.total_rewards[0] += self.OOB_REWARD
            return self.OOB_REWARD
        for s in self.extra_obstacles:
            if sg.circle_line_segment_intersection(new_pos, e.radius, s.start.as_tuple(), s.end.as_tuple(), False):
                # actual intersection
                e.pos = old_pos
                e.velocity *= 0
                self.total_rewards[0] += self.OOB_REWARD
                return self.OOB_REWARD
                break
            pass
        curr_reward = self.SINGLE_STEP_REWARD 

        for index in np.arange(len(indices_of_close_points))[indices_of_close_points]:
            food = self.stationary_entities[index]
            if np.linalg.norm(food.pos - e.pos) <= food.radius + e.radius:
                # Is touching food, so can eat it
                if food.type == EntityType.FOOD and not food.is_dead:
                    food.make_dead()
                    curr_reward += self.rewards[food.type]
                    # self.stationary_entities.pop(index)
                    self.score[index_of_e] += 1
                else:
                    pass
                    # todo later for more entity types
        
        # now only pop
        for i in range(len(self.stationary_entities)-1, -1, -1):
            if (self.stationary_entities[i].is_dead):
                self.stationary_entities.pop(i)
        
        self.total_rewards[index_of_e] += curr_reward 
        
        return curr_reward

    
    def _outside(self, pos: np.array, gap=0) -> bool:
        "Is this position (with a radius gap) outside the boundaries"
        return np.any(pos - gap < 0) or np.any(pos+gap >= self.size)

    @staticmethod
    def basic_simul(state_rep=None, num_food=200, num_obs=10) -> "SingleAgentSimulation":
        if state_rep is None:
            state_rep = StateRep.StateRepSectorsWithDistAndVelocity()
        foods = []
        for i in range(num_food):
            v = np.random.rand(2)
            c1 = np.array([0.4, 0.4])
            c2 = np.array([0.6, 0.6])
            while np.all(np.logical_and(c1 <= v, v <= c2)):
                v = np.random.rand(2)
            foods.append(Entity(v * 1023, EntityType.FOOD, 5))
        obs = SingleAgentSimulation.get_new_rectangle_obs(num_obs=num_obs)
        # obstacles
        return SingleAgentSimulation([Entity(np.array([500, 500]), EntityType.PREY, 10)],foods , state_rep=state_rep, extra_obstacles=obs)

    @staticmethod
    def get_rectangle_from_line(segment: sg.Segment2, dist_orth:float):
        l2 = dist_orth
        vec = segment.get_vector()
        
        vec /= np.linalg.norm(vec)
        # orthogonal
        orth = np.array([-vec[1], vec[0]])
        orth *= l2
        par_seg = segment.add(orth)
        segs = [ segment, par_seg,
                    sg.Segment2(
                        segment.start,
                        sg.Point2(segment.start.x + orth[0], segment.start.y + orth[1])
                    ),
                    sg.Segment2(
                        segment.end,
                        sg.Point2(segment.end.x + orth[0], segment.end.y+ orth[1])
                    )]
        return segs


    @staticmethod
    def get_new_rectangle_obs(num_obs):
        obs = []
        def overlap(segs: sg.Segment2):
            for seg in segs:
                for i in obs:
                    if sg.intersection(seg, i): return True
            if sg.circ_inside_rectangle(500, 500, 10, segs): return True
            return False
        for i in range(num_obs):
            def get_segs():
                p1 = np.random.rand(2) * 1023
                angle = np.random.rand() * np.pi * 2
                N = 100; M = 50
                dist = np.random.rand() * N + M  #between 100 and 300 length
                l2 = np.random.rand() * N + M
                diff = np.array([np.cos(angle), np.sin(angle)]) * dist
                p2 = p1 + diff
                segment = sg.Segment2(
                    sg.Point2(p1[0], p1[1]),
                    sg.Point2(p2[0], p2[1]),
                )
                segments = SingleAgentSimulation.get_rectangle_from_line(segment, l2)

                return segments
            segments = get_segs()
            while overlap(segments):
                segments = get_segs()
            obs.extend(segments)
        return obs
        

    @staticmethod
    def get_food(n=100):
        foods = []
        for i in range(n):
            v = np.random.rand(2)
            c1 = np.array([0.2, 0.2])
            c2 = np.array([0.8, 0.8])
            while np.all(np.logical_and(c1 <= v, v <= c2)):
                v = np.random.rand(2)
            foods.append(Entity(v * 1023, EntityType.FOOD, 5))
        return foods
        
    @staticmethod
    def get_evaluation_level() -> "SingleAgentSimulation":
        state_rep = StateRep.StateRepSectorsWithDistAndVelocity()
        # get food

        # get obstacles
        # make some sort of rectangle thing.
        p1 = sg.Point2(1024/2, 1024/4)
        p2 = sg.Point2(1024/4, 1024/2)

        p3 = sg.Point2(1024/2, 1024/4)
        p4 = sg.Point2(1024/4*3, 1024/2)

        p5 = sg.Point2(0, 1024/2)
        p6 = sg.Point2(1024, 1024/2)
        segments = [
            sg.Segment2(p1, p2),
            sg.Segment2(p3, p4),
            # sg.Segment2(p5, p6),
        ]
        foods = []
        return SingleAgentSimulation([Entity(np.array([500, 500]), EntityType.PREY, 10)], SingleAgentSimulation.get_food(200), state_rep=state_rep, extra_obstacles=segments)
        pass
    
    @staticmethod
    def obstacle_avoidance_level() -> "SingleAgentSimulation":
        state = np.random.get_state()
        np.random.seed(42)
        ans = SingleAgentSimulation.basic_simul(StateRep.StateRepSectorsWithDistAndVelocity(), 0, 15)
        np.random.set_state(state)
        return ans
        
    @staticmethod
    def food_finding_level() -> "SingleAgentSimulation":
        state = np.random.get_state()
        np.random.seed(42)
        ans = SingleAgentSimulation.basic_simul(StateRep.StateRepSectorsWithDistAndVelocity(), 200, 0)
        np.random.set_state(state)
        return ans
    
    @staticmethod
    def combination_hard_coded_eval_level() -> "SingleAgentSimulation":
        s = SingleAgentSimulation.SIZE;
        l = 300
        # corridor width
        cl = 100
        c = s // 2
        sg1 = sg.Segment2(
            sg.Point2(c + cl/2, c + cl/2),
            sg.Point2(c + cl/2 + l, c + cl/2)
        )
        sg2 = sg.Segment2(
            sg.Point2(c - cl/2, c - cl/2),
            sg.Point2(c - cl/2 - l, c - cl/2)
        )
        sg3 = sg.Segment2(
            sg.Point2(c - cl/2, c + cl/2 + l),
            sg.Point2(c - cl/2 - l, c + cl/2 + l)
        )
        sg4 = sg.Segment2(
            sg.Point2(c + cl/2, c - cl/2 - l),
            sg.Point2(c + cl/2 + l, c - cl/2 - l)
        )
        segs = []
        for s in [sg1,sg2,sg3, sg4]:
            segs.extend(SingleAgentSimulation.get_rectangle_from_line(s, l))
        
        # and now food pieces
        n_per_row = 8
        xs = [c + cl/2 + l + cl, c + cl/2 + l + cl,
              c - cl/2 - l - cl, c - cl/2 - l - cl]
        ys = [c + cl/2 + l + cl/2, c - cl/2 - cl/2, 
              c + cl/2 + l + cl/2, c - cl/2 - cl/2]
        foods = []
        for x, y in zip(xs, ys):
            tmpy = y
            dnorm = cl/2
            for i in range(n_per_row):
                dx = np.sin(np.pi * 2 * i/n_per_row) * cl/2
                dy = dx

                foods.append(Entity(np.array([x+dx, tmpy]), EntityType.FOOD, 5))
                foods.append(Entity(np.array([tmpy, x]), EntityType.FOOD, 5))
                tmpy -= dnorm
        # foods = [Entity(np.array([x, y]), EntityType.FOOD, 10)]
        return SingleAgentSimulation([Entity(np.array([500, 500]), EntityType.PREY, 10)], foods , state_rep=StateRep.StateRepSectorsWithDistAndVelocity(),
         extra_obstacles=segs)
    @staticmethod
    def fixed_random_eval_level(seed: int) -> "SingleAgentSimulation":
        """
        This returns a random simulation with the given seed.
        It does not change the global numpy random seed.
        """
        state = np.random.get_state()
        np.random.seed(seed)
        ans = SingleAgentSimulation.basic_simul(StateRep.StateRepSectorsWithDistAndVelocity())
        np.random.set_state(state)
        return ans;
        pass

class SimulationRunner:
    def __init__(self, simulation: SingleAgentSimulation, agent: Agent):
        self.simulation = simulation
        self.agent = agent
    
    def update(self):
        curr_state, indices = self.simulation._get_state(0)
        action = self.agent.get_action(curr_state)
        reward = self.simulation.update(action, indices)
        if isinstance(self.agent, SimpleQLearningAgent) or isinstance(self.agent, ERLAgent):
            self.agent.add_sample(curr_state, action, reward, self.simulation._get_state(0)[0])
        else:
            self.agent.add_sample(curr_state, action, reward)
        pass
    