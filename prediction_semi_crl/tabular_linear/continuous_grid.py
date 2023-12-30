import numpy as np
from gym import core, spaces

class continuousGrid():
    def __init__(self, goal_threshold=0.1, noise_step=0.01, noise_reward=0.0, thrust=0.1):
        self.noise_step = noise_step
        self.goal_threshold = goal_threshold
        self.noise_reward = noise_reward
        self.action_space = spaces.Discrete(4)
        self.thrust = thrust
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.goal = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        self.goal_reward = [1, -1, -1, 1]
        self.reward = 0.0

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, state=None):
        currentcell = state if state is not None else np.random.uniform(low=0.45, high=0.55, size=(2,))
        self.state = currentcell
        return self.state
    
    def step(self, action):
        reward = self.reward
        self.state += self.directions[action] * self.thrust + np.random.uniform(low=-self.noise_step, high=self.noise_step, size=(2,))
        self.state = np.clip(self.state, 0.0, 1.0)
        done_list = [np.linalg.norm((self.state - goal), ord=1) < self.goal_threshold for goal in self.goal] 
        done = any(done_list)
        if done:
            idx = [i for i, x in enumerate(done_list) if x][0]
            reward = np.random.randn() * self.noise_reward + self.goal_reward[idx]
        return self.state, reward, done, None