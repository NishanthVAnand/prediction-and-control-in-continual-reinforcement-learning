import numpy as np

from gym import spaces
from scipy import signal

class discreteGrid():
    def __init__(self, n=7):
        self.n = n
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.n), spaces.Discrete(self.n)))
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.goal = [(0, 0), (0, self.n-1), (self.n-1, 0), (self.n-1, self.n-1)]
        self.start = ((self.n-1)//2, (self.n-1)//2)
        self.goal_reward = {(0, 0):1, (0, self.n-1):-1, (self.n-1, 0):-1, (self.n-1, self.n-1):1}
        self.reward = 0.0

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def get_D(self):
        state_to_idx = lambda x: x[1] * self.n + x[0]
        h_pi = np.zeros(self.n**2)
        h_pi[state_to_idx(self.start)] = 1
        d_pi = np.linalg.inv(np.eye(self.n**2) - self.P_pi.T).dot(h_pi)
        D = np.diag(d_pi/sum(d_pi))
        return D

    def reset(self, state=None):
        self.currentcell = self.start if state is None else state
        return self.currentcell

    def step(self, action):
        reward = 0
        done = 0

        nextcell = tuple(self.currentcell + self.directions[action])
        if self.observation_space.contains(nextcell):
            self.currentcell = nextcell
        state = self.currentcell
        
        if state in self.goal:
            reward = self.goal_reward[state]
            done = 1
        else:
            reward = self.reward

        return state, reward, done, None

    def get_true_values(self, policy, gamma):
        P = np.zeros((self.action_space.n, self.n ** 2, self.n ** 2))
        R = np.zeros((self.action_space.n, self.n ** 2))
        for y_id in range(self.n):
            for x_id in range(self.n):
                cs = y_id * self.n + x_id
                currentcell = (x_id, y_id)
                for a in range(self.action_space.n):
                    if (x_id, y_id) in self.goal:
                        continue
                    else:
                        new_cell = tuple(currentcell + self.directions[a])
                        if self.observation_space.contains(new_cell):
                            nextcell = new_cell
                        else:
                            nextcell = currentcell
                        ns = nextcell[1] * self.n + nextcell[0]
                        if nextcell in self.goal:
                            R[a, cs] = self.goal_reward[nextcell]
                    P[a, cs, ns] = 1
        self.P_pi = np.einsum("acn, ca -> cn", P, policy)
        self.R_pi = np.einsum("ac, ca -> c", R, policy)
        self.v_pi = np.linalg.inv(np.eye(self.n**2) - gamma * self.P_pi).dot(self.R_pi)
        self.D = self.get_D()
        return self.v_pi