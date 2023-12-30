import numpy as np
from gym import core, spaces

class discreteGrid_v2():
	def __init__(self, n=6, obstacles=None, slippery=0.1):
		self.n = n
		self.action_space = spaces.Discrete(4)
		self.observation_space = spaces.Tuple((spaces.Discrete(self.n), spaces.Discrete(self.n)))
		self.directions = [np.array((-1,0)), np.array((0,-1)), np.array((1,0)), np.array((0,1))]
		self.goal = [(self.n-1, self.n-1), (self.n-1, self.n-2)]
		self.start = (0, 0)
		self.goal_reward = {(self.n-1, self.n-1):1, (self.n-1, self.n-2):-1}
		self.reward = 0.0
		self.obstacles = obstacles if obstacles is not None else [(1, 2), (1, 3), (2,3)]
		self.slippery = slippery

	def seed(self, seed=None):
		np.random.seed(seed)
		return [seed]

	def reset(self, state=None):
		self.currentcell = self.start if state is None else state
		return self.currentcell

	def step(self, action):
		reward = 0
		done = 0

		if self.slippery > np.random.random():
			action = np.random.choice([(action-1)%self.action_space.n, (action+1)%self.action_space.n])
		
		nextcell = tuple(self.currentcell + self.directions[action])
		if self.observation_space.contains(nextcell) and nextcell not in self.obstacles:
			self.currentcell = nextcell
		state = self.currentcell
		
		if state in self.goal:
			reward = self.goal_reward[state]
			done = 1
		else:
			reward = self.reward

		return state, reward, done, None