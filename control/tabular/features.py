import numpy as np

class discreteGrid_tabular:
	def __init__(self, n, goal_states, n_actions=None):
		self.n = n
		self.n_actions = n_actions
		self.goal = goal_states

	def features(self, x, action):
		x = tuple(list(x) + [action])
		f = np.zeros((self.n, self.n, self.n_actions))
		if x not in self.goal:
			f[x] = 1
		return f.flatten()