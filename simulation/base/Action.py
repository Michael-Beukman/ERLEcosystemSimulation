
class Action:
    """This represents a single action. """
    
    @classmethod
    def sample(cls: "Action") -> "Action":
        """ Return a randomly sampled action """
        raise NotImplementedError