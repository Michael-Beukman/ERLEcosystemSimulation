import numpy as np

class State:
    """ A single state. Consists of a numpy array """
    def __init__(self, state: np.array):
        self.state = state
    
    def get_value(self) -> np.array:
        return self.state
