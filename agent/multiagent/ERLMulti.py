from simulation.main.Entity import EntityType
# from simulation.base.Agent import Agent
from simulation.multi_agent_sim.MultiAgent import MultiAgent
from simulation.concrete.SimulDiscreteAction import SimulDiscreteAction
from simulation.base.State import State
import numpy as np
import copy;

class ERLMulti(MultiAgent):
    def __init__(self, num_obs, num_actions, do_learning=True) -> None:
        self.weights: np.array = np.random.randn(num_actions,  num_obs)
        self.action_values = None
        self.alpha = 0.005
        self.gamma = 0.99
        self.step_count = 0
        self.learn = do_learning
        self.rewards = []
        self.eligibility_trace = np.zeros_like(self.weights)
        self.lambd = 0.9
        self.deltas = []
        self.epsilon = 0.1
        self.fitness = 0
        self.num_obs = num_obs
        self.num_actions = num_actions
        # self.learn = False
        super().__init__()

    def get_action(self, s: State) -> SimulDiscreteAction:
        self.epsilon *= 0.99
        self.epsilon = max(self.epsilon, 0.02)
        if self.learn:
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
        if not self.learn: return
        # if reward < 0: print('reward', reward)
        self.rewards.append(reward)
        self.step_count += 1
        # print('valud of a = ', a.value, a)
        a = a.value;
        ss = s.get_value()
        # [_, _, food_up, wall_up, food_right, wall_right, ...] 
        # w * s = V_sa => total expected from here until the episode ends.
        # 'correct' total reward = R_t + max(V_s(t+1)_a)
        # V_sa = R_t + (R_t+1 + R_t+2 ... )
        # V_sa = R_t + R_(t+1) + (V_s_t+2)
        # V_sa = R_t + R_(t+1)  + R_(t+2) ... + (V_s_t+3)
        #  R_t + (V_s_t+1) - V(s_a)
        what_i_think_of_this_state = np.max(np.dot(self.weights, s_next.get_value()))
        delta = reward + self.gamma * what_i_think_of_this_state - self.weights[a].dot(ss)
        self.deltas.append(delta)
        # self.weights[a] += self.alpha * (reward - self.weights[a].dot(ss)) * ss
        self.eligibility_trace *= self.lambd * self.gamma
        # add the features
        self.eligibility_trace[a] += ss
        # self.weights[a] += self.alpha * (delta) * self.eligibility_trace[a]
        update = self.alpha * delta * self.eligibility_trace
        n1 = np.linalg.norm(update)
        # if n1 > 10:update = update / n1 * 10
        self.weights += update
        n = (np.linalg.norm(self.weights))
        if n > 100:
            # print("Bigger", n)
            # print(self.weights)
            self.weights = self.weights / n * 100
            # print(self.weights)
        if self.step_count % 100 == 0:
            pass
            # print(n, 'reward = ', np.mean(self.rewards[-100:]), 'delta=', np.mean(self.deltas[-100:]))
            # print(self.weights)
        return super().add_sample(s, a, reward, s_next)
    
    def reproduce(self) -> "MultiAgent":
        ans = copy.deepcopy(self)
        ans.eligibility_trace *= 0
        ans.mutate(0.2)
        return ans
    
    def mutate(self, p=0.01):
        diff = 0.2
        indices = np.random.rand(*self.weights.shape) <= p
        self.weights[indices] += diff * np.random.randn(*self.weights[indices].shape)
    