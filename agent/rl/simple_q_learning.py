from simulation.base.Agent import  Agent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
# from torch import nn
# import torch
import numpy as np

class SimpleQLearningAgent(Agent):
    """
    This is a simple Q learning agent. 
    It uses eligibility traces and an identity function approximation (i.e. it performs w dot state = value).

    If we have A actions and a S-dimensional state called s, then we have a A x S weight matrix (w).

    To choose an action, we do temp = w @ s => gives us Ax1 vector.
    Then we simply choose the action with the largest value.
    We do randomly choose a different one with probability epsilon.
    """
    def __init__(self, num_obs: int, num_actions: int, 
                       do_learning=True,
                       alpha=0.005, gamma=0.99, lambd=0.9,
                       eps_init=0.1, eps_min=0.02, eps_decay=0.999) -> None:
        """Constructor of Simple Q learning

        Args:
            num_obs (int): number of state dimensions = S in the above example
            num_actions (int): A, the number of unique actions to perform
            do_learning (bool, optional): If this is false, then this method's weights doesn't get updated. Useful for evaluation. Defaults to True.
            alpha (float, optional): Learning rate. Defaults to 0.005.
            gamma (float, optional): Reward discount factor. Defaults to 0.99.
            lambd (float, optional): This controls how much bootstrapping to do. lambd = 1 will be equivalent to monte carlo search, lambda = 0 will be equivalent to 0 step TD. Defaults to 0.9.
            eps_init (float, optional): The starting value of epsilon. Defaults to 0.1.
            eps_min (float, optional): The minimum value of epsilon. Defaults to 0.02.
            eps_decay (float, optional): How much epsilon should decay every step. A non zero value makes the agent gradually become more deterministic. Defaults to 0.999.
        """
        self.weights: np.array = np.random.randn(num_actions,  num_obs)
        self.action_values = None
        self.alpha = alpha
        self.gamma = gamma
        self.step_count = 0
        self.learn = do_learning
        self.rewards = []
        self.eligibility_trace: np.array = np.zeros_like(self.weights)
        self.lambd = lambd
        self.deltas = []
        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = self.eps_init
        super().__init__()

    def get_action(self, s: State) -> SimulDiscreteAction:
        """This returns an action based on our weights, given the state

        Args:
            s (State): The current environment state

        Returns:
            SimulDiscreteAction: What action to perform
        """
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.epsilon, self.eps_min)
        # Randomly choose an action epsilon fraction of the time.
        if np.random.rand() < self.epsilon:
            return SimulDiscreteAction.sample()

        # 8 x 18 dot (18x1) => 8 x 1
        
        # This represents the value we expect to get when performing each action in this state.
        self.action_values = np.dot(self.weights, s.get_value())
        
        # Choose the best one. This line ensures if multiple actions have the same maximum value, then 
        # we choose one at random instead of biasing towards the first action with that maximum value.
        best_indices = np.arange(len(self.action_values))[self.action_values == self.action_values.max()]
        action_to_take = np.random.choice(best_indices)
        return SimulDiscreteAction(action_to_take)

    def add_sample(self, s: State, a: SimulDiscreteAction, reward: float, s_next: State):
        """Updates the weights according to Sutton and Barto, 2018, chapter 6.5 and chapter 12.
        Uses Q learning with eligibility traces

        Args:
            s (State): The state this agent just took an action in
            a (SimulDiscreteAction): The action the agent took
            reward (float): The reward obtained using this action
            s_next (State): The next state

        """
        if not self.learn: return
        self.rewards.append(reward)
        self.step_count += 1
        a = a.value;
        ss = s.get_value()
        # [_, _, food_up, wall_up, food_right, wall_right, ...] 

        # This is max_a{Q(S', a)}
        what_i_think_of_this_state = np.max(np.dot(self.weights, s_next.get_value()))
        # self.weights[a].dot(ss) is Q(S, A)
        # Standard delta.
        delta = reward + self.gamma * what_i_think_of_this_state - self.weights[a].dot(ss)
        self.deltas.append(delta)
        
        # decay traces
        self.eligibility_trace *= self.lambd * self.gamma
        
        # add the features to the traces. The features are actually grad{v(S, w)}
        # only update the action.
        self.eligibility_trace[a] += ss
        # Update = zt * deltat
        # weights += update * alpha
        update = self.alpha * delta * self.eligibility_trace
        self.weights += update
        n = (np.linalg.norm(self.weights))
        # Ensure weight magnitudes are manageable and the don't overflow
        if n > 100:
            self.weights = self.weights / n * 100
        if self.step_count % 100 == 0:
            pass
        return super().add_sample(s, a, reward)