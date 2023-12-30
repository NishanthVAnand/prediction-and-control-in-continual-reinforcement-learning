import gym
from discrete_grid_v2 import discreteGrid_v2

import numpy as np

class DiscreteGrid_v2_CL(discreteGrid_v2):
	def __init__(self, n, obstacles=None, goal_reward=None):
		super().__init__(n)
		if goal_reward is not None:
			self.goal_reward = {(self.n-1, self.n-1):goal_reward[0], (self.n-1, self.n-2):goal_reward[1]}			
		else:
			goal_reward = super().goal_reward

def make_env(env_name, misc_param):
	all_envs = []
	if env_name == "DiscreteGrid-v2":
		env_params = [(1.0, -1.0), (-1.0, 1.0)]
		for wiggle in env_params:
			all_envs.append(DiscreteGrid_v2_CL(n=int(misc_param['size']), goal_reward=wiggle))

	else:
		raise NotImplementedError

	return all_envs