from typing import List, Tuple
from simulation.main.Entity import Entity, EntityType
# from simulation.base.Agent import Agent
from simulation.multi_agent_sim.MultiAgent import MultiAgent
from simulation.concrete.StateRepresentations import StateRepresentation
import numpy as np
from simulation.base.State import State

import simulation.utils.utils as sg
from simulation.multi_agent_sim.hardcode_agents.HardCodeAgents import HardCodePred, HardCodePrey, RandomMultiAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from timeit import default_timer as tmr
from agent.multiagent.ERLMulti import ERLMulti
class MultiAgentEntity(Entity):
    """
    Don't use this. This is the old, deprecated multi agent sim. It's very slow for multiple agents. 
    This is just here for completeness.
    Rather use the DiscreteMultiAgentSimulation.
    """
    START_ENERGY = 700
    MAX_ENERGY = 1000 #might need to make this dynamic
    def __init__(self, pos: np.array, type: EntityType, radius: float):
        super().__init__(pos, type, radius=radius)
        self.lifetimer = 0
        self.energy = MultiAgentEntity.START_ENERGY

    def update(self, acc: np.array, dt: float):
        # update position and velocity
        super().update(acc, dt)
        
        # life things
        self.lifetimer += 1
        self.energy -= 1

        # die if you are too old or too hungry.
        if self.lifetimer > MultiAgentSimulation.entity_lifetimes[self.type]:
            self.make_dead()
        
        if self.energy <= 0:
            # print( self.id , " id dying.")
            # just for now.
            self.make_dead()

    
    def eat(self, other: "MultiAgentEntity"):
        # todo
        if other.type == EntityType.FOOD:
            self.energy += 100
        else:
            self.energy += 250

        pass

    @staticmethod
    def food(pos):
        return MultiAgentEntity(pos, EntityType.FOOD, 3)
        pass

    def copy(self) -> "MultiAgentEntity":
        return MultiAgentEntity(self.pos.copy(), self.type, self.radius)

    def __repr__(self) -> str:
        return f"{self.type}, ({self.pos[0], self.pos[1]})"

class MultiAgentSimulation:
    # todo
    entity_lifetimes = {
        EntityType.FOOD: float('inf'),
        EntityType.PREY: 4000 * 0.4, # float('inf'), # 4000*100,
        EntityType.PREDATOR: 4000 * 0.5 #float('inf'), #4000*100,
    }

    entity_energy_to_reproduce = {
        EntityType.PREY: 800,
        EntityType.PREDATOR: 2000 
    }
    masses = {
        EntityType.PREY: 1,
        EntityType.PREDATOR: 3,
    }
    
    OOB_REWARD = -100 # out of bounds
    SINGLE_STEP_REWARD = 0
    DIE_REWARD = -1000
    VIZ_DIST = 50
    SIZE = 1240
    #SIZE = 1920
    MAX_FOOD = 1000

    def __init__(self, state_rep: StateRepresentation, 
                    agent_preds: List[MultiAgent],
                    agent_prey: List[MultiAgent],
                    
                    ent_preds:List[MultiAgentEntity],
                    ent_prey:List[MultiAgentEntity],
                    foods: List[MultiAgentEntity],
                ):
        self.size = MultiAgentSimulation.SIZE
        self.e_predators: List[MultiAgentEntity] = ent_preds
        self.e_prey: List[MultiAgentEntity] = ent_prey
        
        self.a_predators: List[MultiAgent] = agent_preds
        self.a_prey: List[MultiAgent] = agent_prey

        self.foods: List[MultiAgentEntity] = foods
        self.state_rep: StateRepresentation = state_rep

        # entity.id -> state
        self.previous_states = {}

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
            EntityType.FOOD: 1,
            EntityType.PREY: 10
        }

        self.dt = 0.1


        self.current_step_reward = []
        # todo
        extra_obstacles = []

        self.obstacles: List[sg.Segment2] = self.get_obstacles(extra_obstacles)
        self.extra_obstacles = extra_obstacles if extra_obstacles is not None else []
        self.prev_time = tmr()

    def get_obstacles(self, extra_obstacles) -> List[sg.Segment2]:
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
    @property
    def smart_entities(self) -> List[MultiAgentEntity]:
        return self.e_prey + self.e_predators
        pass
    
    @property
    def agents(self) -> List[MultiAgent]:
        return self.a_prey + self.a_predators

    @staticmethod
    def prune(li1: List[MultiAgentEntity], li2: List[MultiAgent]) -> None:
        is_li_2_good = li2 is not None
        for index in range(len(li1)-1, -1, -1):
            # if is_li_2_good:print("Is dead: ", li1[index].is_dead)
            if li1[index].is_dead:
                li1.pop(index)
                if is_li_2_good:
                    li2.pop(index)
    
    

    def update(self):
        self.dt = tmr() - self.prev_time
        self.prev_time = tmr()
        # print(self.dt)
        """
        This performs a single step. It gets actions from each agents and adds a sample.
        """
        # todo mike.
        actions = []
        states = []
        inds = []
        rewards = [0 for _ in (self.smart_entities)]
        # print('ree here coen', len(self.smart_entities))
        # todo potentially later add functionality to remove agents when they die.
        self.current_step_reward = [0 for i in (self.smart_entities)]
        for i in range(len(self.smart_entities)):
            state, indices = self._get_state(i)
            agent_action = self.agents[i].get_action(state)
            actions.append(agent_action)
            states.append(state)
            inds.append(indices)

        for i, action in enumerate(actions):
            rewards[i] += self._perform_action(self.smart_entities[i], action, inds[i])
        
        for  i, action in enumerate(actions):
            # print('index ree',i)
            e = self.smart_entities[i] 
            # if it died by not being old, then negative reward.
            if e.is_dead and e.lifetimer < self.entity_lifetimes[e.type]:
                rewards[i] += MultiAgentSimulation.DIE_REWARD
            # todo optimise
            self.agents[i].add_sample(states[i], action, rewards[i], self._get_state(i)[0])
        # print('pruning')
        # now prune dead agents:
        MultiAgentSimulation.prune(self.e_predators, self.a_predators)
        MultiAgentSimulation.prune(self.e_prey, self.a_prey)
        MultiAgentSimulation.prune(self.foods, None)

        self.grow_food()
        self.reproduce()
        # print('done')
        # self.prev_time = tmr()
        # print(f"Time for one update = {round((ee - ss)*1000)}ms")
    def grow_food(self):
        # print("Energy: ", self.e_prey[0].energy)
        # some sort of carrying capacity growth rate thing.
        rando = np.random.rand(len(self.foods))
        r = 0.01
        index = rando < r * (1 - len(self.foods) / MultiAgentSimulation. MAX_FOOD)
        for i in np.arange(len(self.foods))[index]:
            # v = (np.random.rand(2) * 2 - 1) * 100
            # k = 5
            # v[(v < k) & (v >= 0)] = k
            # v[(v > -k) & (v <= 0)] = -k
            # pos = np.clip(v + self.foods[i].pos, 0, MultiAgentSimulation.SIZE)
            pos = np.floor(np.random.rand(2) * MultiAgentSimulation.SIZE)
            self.foods.append(MultiAgentEntity.food(pos))

        if np.random.rand() < 1 * (1 - len(self.foods) / MultiAgentSimulation. MAX_FOOD):
        #     # print('adding')
            self.foods.append(MultiAgentEntity.food(np.floor(np.random.rand(2) * MultiAgentSimulation.SIZE)))
        pass

    def reproduce(self):
        """
        This goes through all of the entities and reproduces if they have enough energy.
        """
        
        v1s = [self.e_prey, self.e_predators]
        v2s = [self.a_prey, self.a_predators]
        for v1, v2 in zip(v1s, v2s):
            new_agents = []
            new_ents = []
            for i, e in enumerate(v1):
                if e.energy >= self.entity_energy_to_reproduce[e.type]:
                    # print("entities ")
                    new_agent = v2[i].reproduce()
                    new_entity = v1[i].copy()
                    dir = np.array([1, 0])
                    dist = (new_entity.radius + e.radius) * 1.2
                    if self._outside(new_entity.pos + dir * dist):
                        dir *= -1
                    new_entity.pos += dir * dist
                    
                    new_agents.append(new_agent)
                    new_ents.append(new_entity)
                    e.energy = MultiAgentEntity.START_ENERGY / 2 #  self.entity_energy_to_reproduce[e.type] - 5
                # else:print('e=', e.energy)
            v1.extend(new_ents)
            v2.extend(new_agents)
    
    def _get_state(self, index: int) -> Tuple[State, np.array]:
        return self.state_rep.get_state(index, self)

    def get_all_entities(self):
        return self.foods + self.smart_entities
        pass
        
    def _outside(self, pos: np.array, gap=0) -> bool:
        return np.any(pos - gap < 0) or np.any(pos+gap >= self.size)

    def _perform_action(self, ent: MultiAgentEntity, action: SimulDiscreteAction, indices_of_close_points: np.array):
        if ent.is_dead: 
            return self.DIE_REWARD
        
        indices_of_close_points = indices_of_close_points
        index_of_e = self.smart_entities.index(ent)
        acc = self.velocities[action] * 20 / MultiAgentSimulation.masses[ent.type]
        old_pos = ent.pos.copy()
        ent.update(acc, self.dt / 5 * 5)
        new_pos = ent.pos
        # todo remwove
        # ent.pos = old_pos;
        # Am I out of bounds
        if (self._outside(new_pos, ent.radius)):
            ent.pos = old_pos
            ent.velocity *= 0
            self.current_step_reward[index_of_e] += self.OOB_REWARD
            return self.current_step_reward[index_of_e]
        
        # Am I hitting an obstacle TODO
        for s in self.extra_obstacles:
            if sg.circle_line_segment_intersection(new_pos, ent.radius, s.start.as_tuple(), s.end.as_tuple(), False):
                # actual intersection
                ent.pos = old_pos
                ent.velocity *= 0
                # todo not sure if we should return here?
                self.current_step_reward[index_of_e] += self.OOB_REWARD
                return self.current_step_reward[index_of_e]
        # base level reward.
        self.current_step_reward[index_of_e] += self.SINGLE_STEP_REWARD 

        # Now we go over the entities that this might touch.
        all_entites = self.get_all_entities()
        for index in np.arange(len(indices_of_close_points))[indices_of_close_points]:
            newe = all_entites[index]
            # don't interact with dead agent.
            if newe.is_dead:
                continue

            if np.linalg.norm(newe.pos - ent.pos) <= newe.radius + ent.radius:
                # Is touching food, so some interaction can take place.
                if ent.type == EntityType.PREY:
                    if newe.type == EntityType.PREY:
                        # no interaction with prey
                        pass
                    elif newe.type == EntityType.PREDATOR:
                        index_of_pred = self.smart_entities.index(newe)
                        # Do we need to do this, or can the predator eat in there step?
                        # index_of_pred_in_
                        # get eaten. How do we handle rewards????
                        self.current_step_reward[index_of_e] += self.DIE_REWARD
                        self.current_step_reward[index_of_pred] += self.rewards[ent.type]
                        newe.eat(ent)
                        # print("Making prey dead bc pred")
                        ent.make_dead()

                    elif newe.type == EntityType.FOOD:
                        # eat
                        ent.eat(newe)
                        # print("Making food dead bc prey")
                        newe.make_dead()
                        self.current_step_reward[index_of_e] += self.rewards[newe.type]

                elif ent.type == EntityType.PREDATOR:
                    #predator does not interact with preds or food (grass).

                    if newe.type == EntityType.PREY:
                        ent.eat(newe)
                        self.current_step_reward[index_of_e] += self.rewards[newe.type]
                        # eat prey
                        pass
                else:
                    assert 1 == 0 # shouldn't be here 
        return self.current_step_reward[index_of_e]
        pass
    
    @staticmethod
    def get_proper_sim_for_learning(state_rep: StateRepresentation, agent_pred='hard', agent_prey='hard', n_preds=10, n_preys=30) -> "MultiAgentSimulation":
        # n_preds = 10
        # n_preys = 30
        pred_pos = [
            # (100, 100),
            # (200, 200),
            # (300, 300),
            # (400, 400),
            # (500, 500),
        ]
        prey_pos = [
            # (600, 600),
            # (700, 700),
            # (800, 800),
            # (900, 900),
            # (1000, 1000),
        ]
        for p in range(n_preds):
            pred_pos.append((np.random.rand(2) * np.array([0.5, 1])) * MultiAgentSimulation.SIZE)
        for p in range(n_preys):
            prey_pos.append((np.random.rand(2) * np.array([0.5, 1]) + np.array([0.5, 0])) * MultiAgentSimulation.SIZE)
        preds = [MultiAgentEntity(np.array(v), EntityType.PREDATOR, 7) for v in pred_pos]
        
        preys = [MultiAgentEntity(np.array(v), EntityType.PREY, 5) for v in prey_pos]
        foods = []
        for i in range(40):
            # break
            foods.append(MultiAgentEntity.food(np.array([i*30, i*30])))

        for i in range(300):
            # break
            foods.append(MultiAgentEntity.food(np.floor(np.random.rand(2) * MultiAgentSimulation.SIZE)))
        
        agent_func_pred = HardCodePred if agent_pred == 'hard' else (RandomMultiAgent if agent_pred == 'random' else lambda : ERLMulti(state_rep.get_num_features(), len(SimulDiscreteAction)))
        agent_func_prey = HardCodePrey if agent_prey == 'hard' else (RandomMultiAgent if agent_prey == 'random' else lambda : ERLMulti(state_rep.get_num_features(), len(SimulDiscreteAction)))

        agent_preds = [ agent_func_pred() for _ in preds ]

        agent_prey = [ agent_func_prey() for _ in preys ]

        return MultiAgentSimulation(
            state_rep, agent_preds, agent_prey, preds, preys, foods
        )
        

    @staticmethod
    def basic_test(state_rep: StateRepresentation) -> "MultiAgentSimulation":
        preds = [
                MultiAgentEntity(np.array([100, 100]), EntityType.PREDATOR, 10), 
                MultiAgentEntity(np.array([600, 600]), EntityType.PREDATOR, 10)
                ]
        
        preys = [
                MultiAgentEntity(np.array([500, 500]), EntityType.PREY, 5), 
                MultiAgentEntity(np.array([600, 500]), EntityType.PREY, 5), 
                MultiAgentEntity(np.array([700, 500]), EntityType.PREY, 5), 
                MultiAgentEntity(np.array([800, 500]), EntityType.PREY, 5), 
                # MultiAgentEntity(np.array([550, 450]), EntityType.PREY, 5)
                ]
        foods = []
        for i in range(40):
            # break
            foods.append(MultiAgentEntity.food(np.array([i*30, i*30])))

        for i in range(100):
            # break
            foods.append(MultiAgentEntity.food(np.floor(np.random.rand(2) * MultiAgentSimulation.SIZE)))
        

        agent_preds = [
            HardCodePred(),
            HardCodePred(),
        ]

        agent_prey = [
            HardCodePrey(),
            HardCodePrey(),
            HardCodePrey(),
            HardCodePrey(),
            # HardCodePrey(),
        ]

        return MultiAgentSimulation(
            state_rep, agent_preds, agent_prey, preds, preys, foods
        )
        
        pass
