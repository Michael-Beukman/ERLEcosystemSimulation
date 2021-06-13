from simulation.base.Agent import  Agent
from simulation.base.State import State
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
import numpy as np

class ERLAgent(Agent):
    """
    This agent will attempt to combine evolutionary search with genetic algorithms.
    It will learn during its lifetime using RL, and then also perform crossover and mutation across populations.
    This doesn't have a lot of documentation as it is really just EvolutionaryAgent.py and simple_q_learning.py combined.
    """
    def __init__(self, num_obs, num_actions,
                 do_learning=True,
                 alpha=0.005, gamma=0.99, lambd=0.9,
                 eps_init=0.1, eps_min=0.02, eps_decay=0.999) -> None:
        self.weights: np.array = np.random.randn(num_actions,  num_obs)
        self.action_values = None
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

        self.step_count = 0
        self.learn = do_learning
        self.rewards = []
        self.eligibility_trace = np.zeros_like(self.weights)
        self.deltas = []
        self.fitness = 0
        self.num_obs = num_obs
        self.num_actions = num_actions

        self.eps_init = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = self.eps_init
        super().__init__()

    def get_action(self, s: State) -> SimulDiscreteAction:
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.epsilon, self.eps_min)
        if np.random.rand() < self.epsilon:
            return SimulDiscreteAction.sample()
        # if not learning, take best greedy action.
        # 8 x 18 dot (18x1) => 8 x 1
        # w1, w2
        self.action_values = np.dot(self.weights, s.get_value())
        # print(self.action_values.shape)
        best_indices = np.arange(len(self.action_values))[self.action_values == self.action_values.max()]
        # action_to_take = np.argmax()
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
        # ss = [_, _, food_up, wall_up, food_right, wall_right, ...] 

        # This is max_a{Q(S', a)}
        what_i_think_of_this_state = np.max(np.dot(self.weights, s_next.get_value()))
        # self.weights[a].dot(ss) is Q(S, A)
        # Standard delta
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
        # Ensure weight magnitudes are manageable and the don't overflow.
        if n > 100:
            self.weights = self.weights / n * 100
        return super().add_sample(s, a, reward)
    
    def crossover(self, other: "ERLAgent") -> "ERLAgent":
        """
        This performs a crossover operation with another agent and returns a child.
        It basically takes some weights from self and some from others.

        Returns:
            ERLAgent: The child to return
        """
        # Init with same params that I have.
        child = ERLAgent(self.num_obs, self.num_actions, do_learning=True,
                         alpha=self.alpha, gamma=self.gamma, lambd=self.lambd, eps_init=self.eps_init,
                         eps_decay=self.eps_decay, eps_min=self.eps_min)
        # print("BEFORE ", list(child.network.parameters()))
        answer_array = np.zeros_like(self.weights)
        indices = np.random.rand(self.weights.shape[0], self.weights.shape[1]) < 0.5
        answer_array[indices] += self.weights[indices]
        answer_array[~indices] += other.weights[~indices]
        child.weights = answer_array
        return child
    
    def mutate(self, p=0.01):
        """
        This mutates this current individual. Every param is changed with probability p. 
        It randomly adds some value.

        Args:
            p (float, optional): probability. Defaults to 0.01.
        """
        diff = 0.2
        indices = np.random.rand(*self.weights.shape) <= p
        self.weights[indices] += diff * np.random.randn(*self.weights[indices].shape)
    