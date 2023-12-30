import numpy as np
import math

class RBF:
	def __init__(self, order=25, high=None, low=None):
		self.n = order
		self.high = high
		self.low = low
		self.d = len(self.high)
		partition = np.linspace(np.zeros_like(self.low), np.ones_like(self.high), self.n)
		centers = np.meshgrid(*[partition[:,i] for i in range(self.d)])
		self.positions = np.vstack(list(zip(*[centers[i].ravel() for i in range(self.d)]))) 
		self.var_sq = (0.75/(self.n-1))**2
		
	def features(self, x):
		x = (x - self.low)/(self.high - self.low)
		f_x = np.exp(-1/(2*self.var_sq) * np.sum((self.positions-x)**2, axis=1))
		f_x[f_x >= 0.5] = 1.0
		f_x[f_x < 0.5] = 0.0
		return f_x

class discreteGrid_features:
	def __init__(self, n, goal_states):
		self.n = n
		self.goal = goal_states

	def features(self, x):
		x = tuple(x)
		loc_x = np.zeros(self.n)
		loc_y = np.zeros(self.n)
		if x not in self.goal:
			loc_x[x[0]] = 1
			loc_y[x[1]] = 1
		return np.concatenate((loc_x, loc_y))

class discreteGrid_tabular:
	def __init__(self, n, goal_states):
		self.n = n
		self.goal = goal_states

	def features(self, x):
		x = tuple(x)
		f = np.zeros((self.n, self.n))
		if x not in self.goal:
			f[x] = 1
		return f.flatten()