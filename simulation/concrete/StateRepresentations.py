from typing import List
import numpy as np
import simulation.main.Simulation as Sim
from simulation.base.State import State
from simulation.main.Entity import Entity, EntityType
# import skgeom as sg
import simulation.utils.utils as sg

class StateRepresentation:
    """
    This basically takes in a simulation and returns a state for each agent.
    """
    def __init__(self):
        pass

    def get_num_features(self) -> int:
        raise NotImplementedError
    
    def get_state(self, index: int, simulation: "Sim.Simulation") -> np.array:
        raise NotImplementedError

class StateRepSectorsWithDistAndVelocity(StateRepresentation):
    """
    This is a staterepresentation that returns a vector of size 18.
    The first two elements are 0, but are reserved for the agent's velocity if necessary. Not currently in use.
    The next 16 elements are split into 8 groups of 2, which represents the different directions.
    
    each of these are [fd, wd] where fd is 1 - distance to the closest food particle, normalised between 0 and 1. 
    0 means that there is no food there.
    wd is the same as fd, but with wall distance rather than food distance.
    """
    NUM_CONES = 8
    HAS_BIAS = False
    def __init__(self):
        super().__init__()
        self.angle_per_cone = 2 * np.pi / StateRepSectorsWithDistAndVelocity.NUM_CONES
        self.segments = None

    def get_num_features(self) -> int:
        # for each cone, we do the following:
        # VIZ - dist to closest food, VIZ - wall dist. Add to that the velocity
        return self.NUM_CONES * (2) + 2 + StateRepSectorsWithDistAndVelocity.HAS_BIAS

    def get_segments(self, sim: "Sim.SingleAgentSimulation") -> List[sg.Segment2]:
        return sim.obstacles
        pass
    def get_state(self, index: int, simulation: "Sim.SingleAgentSimulation") -> np.array:
        if self.segments is None:
            self.segments = self.get_segments(simulation)
        i = index
        sim = simulation
        myent = sim.smart_entities[i]
        # need to do some sort of raycasting/cone thing
        # array of (distance, angle). Assuming up has angle 0.
        all_positions = []
        all_types = []
        type_list = list(EntityType)
        for e in sim.stationary_entities:
            all_positions.append(e.pos)
            all_types.append((e.type))
        for index, e in enumerate(sim.smart_entities):
            if i == index: continue # don't use self as state
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

        index_where_distance_less = distances <= sim.VIZ_DIST
        distances = distances[index_where_distance_less]

        angles = angles[index_where_distance_less]
        angles[angles < -self.angle_per_cone / 2] += 2 * np.pi
        pos_vecs = diff[index_where_distance_less]
        # all_types = np.array(all_types)[index_where_distance_less]
        # print(pos_vecs)
        # print(np.degrees(angles))



        # now each element has a distance and an angle.
        k = 0
        actual_state = []
        # if it has a bias, add in a 1.
        if StateRepSectorsWithDistAndVelocity.HAS_BIAS:
            actual_state.append(1)
        actual_state += [k * myent.velocity[0] / myent.MAX_VEL, myent.velocity[1] / myent.MAX_VEL * k]
        for i in range(self.NUM_CONES):
            curr_angle = i * self.angle_per_cone - self.angle_per_cone / 2
            v1 = curr_angle <= angles
            v2 = angles < curr_angle + self.angle_per_cone
            indices_in_this_cone = np.logical_and(v1, v2)

            dists_in_cone = distances[indices_in_this_cone]
            
            if len(dists_in_cone) == 0:
                food_dist_val = 0
            else:
                food_dist_val = sim.VIZ_DIST -  np.min(dists_in_cone)
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
            actual_state.extend([food_dist_val / sim.VIZ_DIST, wall_dist_val / sim.VIZ_DIST])

        # print('idx', index_where_distance_less)
        return State(np.array(actual_state) ), index_where_distance_less
