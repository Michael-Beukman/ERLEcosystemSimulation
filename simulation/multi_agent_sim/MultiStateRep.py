from typing import List
from simulation.concrete.StateRepresentations import StateRepresentation
import numpy as np
from simulation.multi_agent_sim.MultiAgentSimulation import MultiAgentSimulation
import simulation.utils.utils as sg
from simulation.main.Entity import EntityType, Entity
from simulation.base.State import State

class MultiAgentStateRep(StateRepresentation):
    NUM_CONES = 8
    def __init__(self):
        super().__init__()
        self.angle_per_cone = 2 * np.pi / MultiAgentStateRep.NUM_CONES
        self.segments = None

    def get_num_features(self) -> int:
        # for each cone, we do the following:
        # VIZ - dist to closest food, VIZ - prey dist, VIZ - pred dist ,VIZ - wall dist and then velocity.
        return self.NUM_CONES * (4) + 2

    def get_segments(self, sim: MultiAgentSimulation) -> List[sg.Segment2]:
        return sim.obstacles

    def get_state(self, index: int, simulation: MultiAgentSimulation) -> np.array:
        if self.segments is None:
            self.segments = self.get_segments(simulation)
        i = index
        sim = simulation
        # print('index ', index, 'length', len(sim.smart_entities))
        myent = sim.smart_entities[i]
        entities_to_do = [EntityType.FOOD, EntityType.PREY, EntityType.PREDATOR]
        # need to do some sort of raycasting/cone thing
        # array of (distance, angle). Assuming up has angle 0.
        all_positions = []
        all_types = []
        for e in sim.get_all_entities():
            all_positions.append(e.pos)
            all_types.append((e.type))
        

        all_positions = np.array(all_positions)
        if len(all_positions):
            diff = all_positions - myent.pos # this is position vectors from center of entity outwards
        else:
            diff = np.array([
                [1000000, 100000]
            ])
        distances = np.linalg.norm(diff, axis=1)
        trans = diff.T
        angles = -np.arctan2(trans[1], trans[0]).T + np.pi/2
        angles[angles > np.pi] -= 2 * np.pi
        angles[angles < -np.pi] += 2 * np.pi

        # this is what is was. Going to try and ignore myself.
        # index_where_distance_less = distances <= sim.VIZ_DIST
        
        index_where_distance_less = np.logical_and(distances <= sim.VIZ_DIST, distances > 0)
        
        distances = distances[index_where_distance_less]
        proper_types = np.array(all_types)[index_where_distance_less]

        angles = angles[index_where_distance_less]
        angles[angles < -self.angle_per_cone / 2] += 2 * np.pi
        # pos_vecs = diff[index_where_distance_less]



        # now each element has a distance and an angle.
        k = 0
        actual_state = [k * myent.velocity[0] / myent.MAX_VEL, myent.velocity[1] / myent.MAX_VEL * k]
        for i in range(self.NUM_CONES):
            
            curr_angle = i * self.angle_per_cone - self.angle_per_cone / 2
            v1 = curr_angle <= angles
            v2 = angles < curr_angle + self.angle_per_cone
            indices_in_this_cone = np.logical_and(v1, v2)

            dists_in_cone = distances[indices_in_this_cone]
            types_in_cone = proper_types[indices_in_this_cone]
            # if index == 1:print(i, len(dists_in_cone), angles)

            list_of_closest_lengths = []
            # print("Ents ", entities_to_do)
            for ent_type in entities_to_do:
                indices_of_this_type = types_in_cone == ent_type
                dists_of_this_type = dists_in_cone[indices_of_this_type]
                if len(dists_of_this_type) != 0:
                    min_dist = np.min(dists_of_this_type)
                else:
                    min_dist = 1000000000
                list_of_closest_lengths.append( max(0, MultiAgentSimulation.VIZ_DIST - min_dist) / MultiAgentSimulation.VIZ_DIST)
            wall_dist_val = 0
            # now check wall collision
            # Check if we shoot out a ray in the middle of our sector, at what distance will we hit a wall?
            
            mid_angle = curr_angle + self.angle_per_cone / 2
            diff = np.array([np.sin(mid_angle) * sim.VIZ_DIST, np.cos(mid_angle) * sim.VIZ_DIST])
            tmp = myent.pos + diff
            
            # start a little bit further 
            to_add = diff / sim.VIZ_DIST * 2
            
            start_point = sg.Point2(myent.pos[0] + to_add[0], myent.pos[1] + to_add[1])

            end_point = sg.Point2(tmp[0], tmp[1])
            seg_mine = sg.Segment2(start_point, end_point)
            # print(mid_angle * 180 / np.pi, seg_mine)
            # s = self.segments[i]
            for s in self.segments:
                inter = sg.intersection(seg_mine, s)
                if inter is not None:
                    # print(inter.x, inter.y, start_point.dist(inter), start_point.x, start_point.y, tmp, i)
                    d = start_point.dist(inter)
                    if d > sim.VIZ_DIST:
                        d = sim.VIZ_DIST
                    wall_dist_val = max(sim.VIZ_DIST - (d), wall_dist_val)
            
            actual_state.extend(list_of_closest_lengths); 
            actual_state.append(wall_dist_val / sim.VIZ_DIST)

        return State(np.array(actual_state) ), index_where_distance_less
