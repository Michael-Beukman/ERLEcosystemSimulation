import numpy as np
from enum import Enum, auto

class EntityType(Enum):
    PREY = auto()
    FOOD = auto()
    PREDATOR = auto()
    NONE = auto()

    def one_hot_encode(self):
        index = dic[self]
        v = np.zeros(len(EntityType))
        v[index] = 1
        return v

li = list(EntityType)
dic = {v: index for index, v in enumerate(li)}

class Entity:
    main_id = 0
    MAX_VEL = 20
    def __init__(self, pos: np.array, type: EntityType, radius:float=1):
        self.id = Entity.main_id
        Entity.main_id += 1
        self.radius = radius
        self.pos = pos.astype(np.float32)
        self.type = type
        self.is_dead = False
        self.velocity = np.zeros_like(pos, dtype=np.float32)
        self.velocity[0] = self.MAX_VEL
        self.acc = np.zeros_like(pos, dtype=np.float32)
    
    def make_dead(self):
        self.is_dead = True
    
    def update(self, acc: np.array, dt:float):
        if acc is None:
            acc = self.acc
        else:
            self.acc = acc
        self.velocity += acc * dt
        n = np.linalg.norm(self.velocity)
        if n > Entity.MAX_VEL:
            self.velocity = self.velocity / n * Entity.MAX_VEL
        self.pos += self.velocity * dt