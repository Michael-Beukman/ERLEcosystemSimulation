from collections import defaultdict
import random
from typing import Dict, List, Tuple
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


class DiscMultiAgentEntity:
    START_ENERGY = 700
    MAX_ENERGY = 1000  # might need to make this dynamic

    def __init__(self, pos: np.array, type: EntityType, radius: float):
        # super().__init__(pos, type, radius=radius)
        self.pos = pos
        self.type = type
        self.radius = radius
        self.lifetimer = 0
        self.energy = DiscMultiAgentEntity.START_ENERGY
        self.is_dead = False

    def make_dead(self):
        self.is_dead = True

    def update(self, motion: np.array):
        # update position and velocity
        self.pos += motion
        # life things
        self.lifetimer += 1
        self.energy -= 1

        # die if you are too old or too hungry.
        if self.lifetimer > DiscreteMultiAgentSimulation.entity_lifetimes[self.type]:
            self.make_dead()

        if self.energy <= 0:
            # print( self.id , " id dying.")
            # just for now.
            self.make_dead()

    def eat(self, other: "DiscMultiAgentEntity"):
        # todo
        if other.type == EntityType.FOOD:
            self.energy += 100
        else:
            self.energy += 250

        pass

    @staticmethod
    def food(pos):
        return DiscMultiAgentEntity(pos, EntityType.FOOD, 3)
        pass

    def copy(self) -> "DiscMultiAgentEntity":
        return DiscMultiAgentEntity(self.pos.copy(), self.type, self.radius)

    def __repr__(self) -> str:
        return f"{self.type}, ({self.pos[0], self.pos[1]})"


class DiscreteMultiAgentSimulation:
    """
    This is the discrete agent simulation, which is basically the single agent sim, 
    but we explicitly support multiple agents, reproduction, energy, etc. 
    This is also discretised instead of continuous for performance reasons.
    """
    entity_lifetimes = {
        EntityType.FOOD: float('inf'),
        EntityType.PREY: 4000 * 0.1,  # float('inf'), # 4000*100,
        EntityType.PREDATOR: 4000 * 0.2  # float('inf'), #4000*100,
    }
    # IWALL = 0
    IFOOD = 0
    IPREY = 1
    IPRED = 2
    IWALL = 3
    DIE_REWARD = -1000
    OOB_REWARD = -100  # out of bounds
    SINGLE_STEP_REWARD = 0
    MAX_FOOD = 1000
    entity_energy_to_reproduce = {
        EntityType.PREY: 800,
        EntityType.PREDATOR: 2000
    }

    def __init__(self,
                 agent_preds: List[MultiAgent],
                 agent_prey: List[MultiAgent],

                 ent_preds: List[DiscMultiAgentEntity],
                 ent_prey: List[DiscMultiAgentEntity],
                 foods: List[DiscMultiAgentEntity],
                 size=1024, food_rate=0.01):
        self.food_rate = food_rate
        self.size = size
        self.current_id = 1
        self.values = np.zeros(
            (4, size, size)
        )
        self.e_predators: Dict[int, DiscMultiAgentEntity] = {}
        self.e_prey: Dict[int, DiscMultiAgentEntity] = {}
        self.a_predators: Dict[int, MultiAgent] = {}
        self.a_prey: Dict[int, MultiAgent] = {}
        self.foods: Dict[int, DiscMultiAgentEntity] = {}

        for ents, agents, mye, mya in zip([ent_preds, ent_prey], [agent_preds, agent_prey],
                                          [self.e_predators, self.e_prey], [self.a_predators, self.a_prey]):
            # print(id(mye))
            # print(ents)
            print("L ", len(ents), len(agents))
            for index in range(len(ents)):
                mye[self.current_id] = ents[index]
                mya[self.current_id] = agents[index]
                self.current_id += 1

        # print(id(self.e_predators))
        # print(id(self.e_prey))
        # print(self.e_predators, self.e_prey, ent_preds, ent_prey); exit()
        for index in range(len(foods)):
            self.foods[self.current_id] = foods[index]
            self.current_id += 1

        for index, dic in enumerate([self.foods, self.e_prey, self.e_predators]):
            for idd, ent in dic.items():
                self.values[index, ent.pos[0], ent.pos[1]] = idd

        # print("BEFORE", self.values.shape)
        self.values: np.ndarray = np.pad(self.values, (
            (0, 0), (1, 1), (1, 1)
        ))
        # print("After", self.values.shape)  # exit()

        self.values[self.IWALL, 0, :] = 1
        self.values[self.IWALL, -1, :] = 1
        self.values[self.IWALL, :, 0] = 1
        self.values[self.IWALL, :, -1] = 1

        self.current_step_reward = defaultdict(lambda: 0),
        self.rewards = {
            EntityType.FOOD: 1,
            EntityType.PREY: 10
        }
        r2 = 1
        self.velocities = {
            SimulDiscreteAction.UP: np.array([0, -1])[::-1],
            SimulDiscreteAction.DOWN: np.array([0, 1])[::-1],
            SimulDiscreteAction.LEFT: np.array([-1, 0])[::-1],
            SimulDiscreteAction.RIGHT: np.array([1, 0])[::-1],

            SimulDiscreteAction.UP_RIGHT: np.array([r2, -r2])[::-1],
            SimulDiscreteAction.DOWN_RIGHT: np.array([r2, r2])[::-1],
            SimulDiscreteAction.DOWN_LEFT: np.array([-r2, r2])[::-1],
            SimulDiscreteAction.UP_LEFT: np.array([-r2, -r2])[::-1],
        }
        self.states = {}
        self.prev_time = tmr()

    def all_entities(self) -> Dict[int, DiscMultiAgentEntity]:
        return {**self.e_predators, **self.e_prey, **self.foods}

    @property
    def smart_entities(self) -> Dict[int, DiscMultiAgentEntity]:
        return {**self.e_prey, **self.e_predators}

    @property
    def smart_agents(self) -> Dict[int, MultiAgent]:
        return {**self.a_prey, **self.a_predators}

    def get_all_entities(self) -> Dict[int, MultiAgent]:
        return {**self.e_prey, **self.e_predators, **self.foods}

    def validate_invariant(self):
        return
        # This checks if all the ids in the dicts are the same as the ids in the np arrays
        # check counts first.
        non_zeros_in_np = np.count_nonzero(self.values[:-1])
        non_zero_in_list = len(self.get_all_entities())
        assert non_zeros_in_np == non_zero_in_list, f"Bad lengths {non_zero_in_list} != {non_zeros_in_np}"
        for id, ent in self.get_all_entities().items():
            if self.vals(self.index_from_type(ent), ent.pos[0], ent.pos[1]) != id:
                assert 1 == 0, f"Bad at position {ent.pos}, with ent = {ent.type} and id = {id}"

    def update(self):
        self.dt = tmr() - self.prev_time
        self.prev_time = tmr()
        # print(self.dt)
        """
        This performs a single step. It gets actions from each agents and adds a sample.
        """
        # todo mike.
        actions = {}
        states = {}
        rewards = {i: 0 for i in self.smart_entities}

        # todo potentially later add functionality to remove agents when they die.
        # self.current_step_reward = [0 for i in (self.smart_entities)]
        agents = self.smart_agents

        for i in self.smart_entities:
            state = self._get_state(i)
            agent_action = agents[i].get_action(state)
            actions[i] = agent_action
            states[i] = state
        self.current_step_reward = defaultdict(lambda: 0)
        acts = list(actions.items())
        # shuffle so we perform actions in a random order
        random.shuffle(acts)
        for i, action in acts:
            rewards[i] += self._perform_action(i,
                                               self.smart_entities[i], action)
        for i, action in actions.items():
            e = self.smart_entities[i]
            # if it died by not being old, then negative reward.
            if e.is_dead and e.lifetimer < self.entity_lifetimes[e.type]:
                rewards[i] += DiscreteMultiAgentSimulation.DIE_REWARD
            # print(states, i)
            agents[i].add_sample(
                states[i], action, rewards[i], self._get_state(i))

        self.prune(self.e_predators, self.a_predators)
        self.prune(self.e_prey, self.a_prey)
        self.prune(self.foods, None)
        self.validate_invariant()

        # TODO DO THESE THINGS
        self.grow_food()
        self.reproduce()
        # print("VALUES ", self.values[self.IPREY].sum())
        # print(self.values[self.IFOOD])
        # print(self.values[self.IPREY])
        # print(self.values[self.IPRED])
        # exit()

    def grow_food(self):
        def test_zeug(og_pos):
            found_pos = False
            for dist in range(10, 20):
                for p in posses:
                    # check if any collisions
                    temp = og_pos + p * dist
                    if self._outside(temp):
                        continue
                    has_any_other_entity = False
                    for index in range(3):
                        if self.vals(index, temp[0], temp[1]) > 0:
                            has_any_other_entity = True
                            break
                    if not has_any_other_entity:
                        # can actually take this position
                        found_pos = True
                        break
                if found_pos:
                    break
            return found_pos, temp

        def add_one(found_pos, temp):
            if found_pos:
                new_entity = DiscMultiAgentEntity.food(temp)
                self.set_vals(self.index_from_type(
                    new_entity), new_entity.pos[0], new_entity.pos[1], self.current_id)
                new_foods[self.current_id] = new_entity
                self.current_id += 1

        posses = self.velocities.values()
        new_foods = {}
        for i in self.foods:
            if np.random.rand() >= self.food_rate * (1 - len(self.foods) / DiscreteMultiAgentSimulation. MAX_FOOD):
                continue
            found_pos, temp = test_zeug(self.foods[i].pos)
            add_one(found_pos, temp)

        if len(self.foods) + len(new_foods) < self.MAX_FOOD:
            for i in range(5):
                found_pos, temp = test_zeug(np.random.randint(
                    low=[0, 0], high=[self.size, self.size]))
                add_one(found_pos, temp)

        for i in new_foods:
            self.foods[i] = new_foods[i]
        pass

    def reproduce(self):
        # print("Before reproduce", len(self.e_predators), len(self.e_prey), len(self.foods))
        # v1s = [self.e_prey, self.e_predators]
        # v2s = [self.a_prey, self.a_predators]
        posses = self.velocities.values()

        for dic_ent, dic_agents in zip([self.e_predators, self.e_prey], [self.a_predators, self.a_prey]):
            new_agents = {}
            new_ents = {}
            for i, e in dic_ent.items():
                if e.energy >= self.entity_energy_to_reproduce[e.type]:
                    # print("Reproducing")
                    # print("entities ")
                    new_agent = dic_agents[i].reproduce()
                    new_entity = dic_ent[i].copy()
                    # todo might collide with something else.
                    found_pos = False
                    for dist in range(1, 10):
                        for p in posses:
                            # check if any collisions
                            temp = new_entity.pos + p * dist
                            if self._outside(temp):
                                continue
                            has_any_other_entity = False
                            for index in range(3):
                                if self.vals(index, temp[0], temp[1]) > 0:
                                    has_any_other_entity = True
                                    break
                            if not has_any_other_entity:
                                # can actually take this position
                                found_pos = True
                                break
                        if found_pos:
                            break
                    # temp pos is proper one!
                    if found_pos:
                        new_entity.pos = temp
                        self.set_vals(self.index_from_type(
                            new_entity), new_entity.pos[0], new_entity.pos[1], self.current_id)
                        new_ents[self.current_id] = new_entity
                        new_agents[self.current_id] = new_agent
                        self.current_id += 1
                        e.energy = DiscMultiAgentEntity.START_ENERGY / 2

            for k in new_ents:
                dic_ent[k] = new_ents[k]
                dic_agents[k] = new_agents[k]
        # print("After reproduce", len(self.e_predators), len(self.e_prey), len(self.foods))

    def vals(self, index, p1, p2):
        # consider padding
        return self.values[index, p1 + 1, p2 + 1]

    def set_vals(self, index, p1, p2, val):
        # consider padding
        self.values[index, p1 + 1, p2 + 1] = val

    @staticmethod
    def index_from_type(ent: DiscMultiAgentEntity):
        if ent.type == EntityType.FOOD:
            return DiscreteMultiAgentSimulation.IFOOD
        return DiscreteMultiAgentSimulation.IPREY if ent.type == EntityType.PREY else DiscreteMultiAgentSimulation.IPRED

    def prune(self, li1: Dict[int, DiscMultiAgentEntity], li2: Dict[int, MultiAgent]) -> None:
        is_li_2_good = li2 is not None
        keys = list(li1.keys())
        for id in keys:
            if li1[id].is_dead:
                ent = li1[id]
                index = self.index_from_type(ent)
                # if it is still on the thing, then remove.
                # if li2 is None:print(f"{id} is a {li1[id].type} that is dead, val in array = ", self.vals(index, ent.pos[0], ent.pos[1]))
                if self.vals(index, ent.pos[0], ent.pos[1]) == id:

                    self.set_vals(index, ent.pos[0], ent.pos[1], 0)

                del li1[id]
                if is_li_2_good:
                    del li2[id]

    def _get_state(self, id_of_agent: int):
        # TODO
        agent = self.all_entities()[id_of_agent]
        if agent.type == EntityType.FOOD:
            return []

        def get_things_in_range(arr, pos, do_wall=False, ver=False):
            # take into account padding
            a, b, c, d = pos[0], pos[0]+3, pos[1], pos[1]+3
            # a, b, c, d = pos[1], pos[1]+3, pos[0], pos[0]+3

            # if do_wall:

            # a = max(a, 0)
            # c = max(c, 0)

            # b = min(b, self.size)
            # d = min(d, self.size)
            thinger = (arr[a:b, c:d])
            if ver:
                print(pos)
                print(arr + self.values[self.IPREY])
                print(thinger)

            # rs= (arr[a:b, c:d] > 0).T.ravel()
            rs = (thinger > 0).ravel()
            rs[4:-1] = rs[5:]
            return rs[:-1]
        # walls
        walls = get_things_in_range(self.values[self.IWALL], agent.pos)

        # food

        # agent.type == EntityType.PREY)
        foods = get_things_in_range(
            self.values[self.IFOOD], agent.pos, ver=False)
        # if agent.type == EntityType.PREY:print(agent.type,foods)
        # # prey
        prey = get_things_in_range(self.values[self.IPREY], agent.pos)
        # # pred
        pred = get_things_in_range(self.values[self.IPRED], agent.pos)

        answer = np.empty((34), dtype=np.float32)
        # velocity dont care about
        answer[0:2] = 0
        answer[2::4] = foods
        answer[3::4] = prey
        answer[4::4] = pred
        answer[5::4] = walls
        self.states[id_of_agent] = answer
        return State(answer)

    def _perform_action(self, id: int, ent: DiscMultiAgentEntity, action: SimulDiscreteAction):
        if ent.is_dead:
            return self.DIE_REWARD

        old_pos = ent.pos.copy()
        motion = self.velocities[action] * 1
        ent.update(motion)
        # print(motion, ent.pos, old_pos)
        new_pos = ent.pos
        if (self._outside(new_pos, 0)):
            # print(new_pos, ' is outside so we go back to: ', old_pos)
            ent.pos = old_pos
            self.current_step_reward[id] += self.OOB_REWARD
            # print("OOB")
            return self.current_step_reward[id]

        # base level reward.
        self.current_step_reward[id] += self.SINGLE_STEP_REWARD

        # for other_ent_id, other_ent in self.smart_entities.items():
        for index in [DiscreteMultiAgentSimulation.IPREY, DiscreteMultiAgentSimulation.IFOOD, DiscreteMultiAgentSimulation.IPRED]:
            other_ent_id = self.vals(index, ent.pos[0], ent.pos[1])

            if other_ent_id == id or other_ent_id == 0:
                continue  # dont compare against self
            other_ent = self.all_entities()[other_ent_id]
            # break
            if other_ent.is_dead:
                continue
            if np.all(other_ent.pos == ent.pos):
                if other_ent.type == ent.type:
                    # Then we cannot move to this pos as it is occupied
                    ent.pos = old_pos
                    self.current_step_reward[id] += self.OOB_REWARD
                    return self.current_step_reward[id]
                else:
                    # Bad things
                    if ent.type == EntityType.PREY:
                        if other_ent.type == EntityType.PREDATOR:
                            # print(f"Pred {other_ent_id} eats Prey {id}")
                            id_of_pred = other_ent_id
                            self.current_step_reward[id] += DiscreteMultiAgentSimulation.DIE_REWARD
                            self.current_step_reward[id_of_pred] += self.rewards[ent.type]
                            other_ent.eat(ent)
                            # print("Making prey dead bc pred")
                            ent.make_dead()
                            # remove from map

                            self.set_vals(DiscreteMultiAgentSimulation.IPREY,
                                          old_pos[0], old_pos[1], 0)
                            return self.current_step_reward[id]
                        else:
                            # has to be food!

                            # eat
                            ent.eat(other_ent)
                            # print("Making food dead bc prey")
                            other_ent.make_dead()
                            self.current_step_reward[id] += self.rewards[other_ent.type]
                            # self.set_vals(DiscreteMultiAgentSimulation.IFOOD,other_ent.pos[0], other_ent.pos[1], 0)

                        pass
                    elif ent.type == EntityType.PREDATOR:
                        if other_ent.type == EntityType.PREY:
                            # print(f"Pred {id} eats Prey {other_ent_id}")
                            ent.eat(other_ent)
                            self.current_step_reward[id] += self.rewards[other_ent.type]
                            other_ent.make_dead()
                            # remove from map.
                            # self.set_vals(DiscreteMultiAgentSimulation.IPREY,other_ent.pos[0], other_ent.pos[1], 0)
                            # eat prey
                            pass

        # TODO at end, if can actually move, then update the pos array!
        # if we got here, we didnt  return yet.
        index = DiscreteMultiAgentSimulation.IPREY if ent.type == EntityType.PREY else DiscreteMultiAgentSimulation.IPRED
        self.set_vals(index, old_pos[0], old_pos[1], 0)
        self.set_vals(index, new_pos[0], new_pos[1], id)
        return self.current_step_reward[id]

    def _outside(self, pos: np.array, gap=0) -> bool:
        return np.any(pos - gap < 0) or np.any(pos+gap >= self.size)

    @staticmethod
    def get_proper_sim_for_learning(size=256,
                                    n_preds={
                                        'erl': 10,
                                        'random': 10,
                                        'hard': 10
                                    }, n_preys={
                                        'erl': 10,
                                        'random': 10,
                                        'hard': 10
                                    }, n_foods=300,
                                    entity_energy_to_reproduce={
                                        EntityType.PREY: 800,
                                        EntityType.PREDATOR: 2000
                                    },
                                    entity_lifetimes={
                                        EntityType.FOOD: float('inf'),
                                        # float('inf'), # 4000*100,
                                        EntityType.PREY: 4000 * 0.1,
                                        # float('inf'), #4000*100,
                                        EntityType.PREDATOR: 4000 * 0.2
                                    },
                                    food_rate=0.01
                                    ) -> "DiscreteMultiAgentSimulation":
        DiscreteMultiAgentSimulation.entity_energy_to_reproduce = entity_energy_to_reproduce
        DiscreteMultiAgentSimulation.entity_lifetimes = entity_lifetimes
        positions = set()
        # n_preds = 1
        # n_preys  = 1
        foods = []
        preds = []
        preys = []
        tot_npreds = sum(n_preds.values())
        tot_npreys = sum(n_preys.values())
        for i in range(n_foods):
            while 1:
                p = tuple(np.random.randint(0, size, size=(2)))
                if p not in positions:
                    positions.add(p)
                    break

            foods.append(DiscMultiAgentEntity.food(np.array(p)))

        for j in range(tot_npreds):
            while 1:
                p = tuple(np.random.randint(0, [size/2, size], size=(2)))
                if p not in positions:
                    positions.add(p)
                    break

            preds.append(DiscMultiAgentEntity(
                np.array(p), EntityType.PREDATOR, 7))

        for j in range(tot_npreys):
            while 1:
                p = tuple(np.random.randint(
                    [size/2, 0], [size, size], size=(2)))
                if p not in positions:
                    positions.add(p)
                    break

            preys.append(DiscMultiAgentEntity(np.array(p), EntityType.PREY, 5))
        
        agent_preds = []
        for agent_pred, count in n_preds.items():
        # if agent_pred == 'random' else lambda : ERLMulti(state_rep.get_num_features(), len(SimulDiscreteAction)))
            agent_func_pred = HardCodePred if agent_pred == 'hard' else (
                RandomMultiAgent if agent_pred == 'random' else lambda: ERLMulti(34, len(SimulDiscreteAction)))
            for c in range(count):
                agent_preds.append(agent_func_pred())
        
        agent_preys = []
        for agent_prey, count in n_preys.items():
        # if agent_pred == 'random' else lambda : ERLMulti(state_rep.get_num_features(), len(SimulDiscreteAction)))
            agent_func_prey = HardCodePrey if agent_prey == 'hard' else (
                RandomMultiAgent if agent_prey == 'random' else lambda: ERLMulti(34, len(SimulDiscreteAction)))
            for c in range(count):
                agent_preys.append(agent_func_prey())
        print("Lengths", len(agent_preys), len(preys), len(agent_preds), len(preds))
        return DiscreteMultiAgentSimulation(
            agent_preds, agent_preys, preds, preys, foods, size=size, food_rate=food_rate
        )


if __name__ == '__main__':
    d = DiscreteMultiAgentSimulation(
        [HardCodePred()],
        [HardCodePrey()],
        [DiscMultiAgentEntity(np.array([0, 0]), EntityType.PREDATOR, 5)],
        [DiscMultiAgentEntity(np.array([3, 3]), EntityType.PREY, 2)],
        [DiscMultiAgentEntity(np.array([2, 1]), EntityType.FOOD, 1)],
        size=5
    )

    print(d.values[0])
    print(d.values[1])
    print(d.values[2])
