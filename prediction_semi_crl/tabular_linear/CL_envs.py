from gym import spaces
from gym.utils import seeding
from continuous_grid import continuousGrid
from discrete_grid import discreteGrid

import numpy as np
import math
from scipy.stats import norm

class ContinuousGrid_CL(continuousGrid):
	def __init__(self, goal_reward=None):
		super().__init__()
		self.goal_reward = goal_reward if goal_reward is not None else super().goal_reward

class DiscreteGrid_CL(discreteGrid):
	def __init__(self, n, goal_reward=None):
		super().__init__(n)
		if goal_reward is not None:
			self.goal_reward = {(0, 0):goal_reward[0], (0, self.n-1):goal_reward[1],\
				 (self.n-1, 0):goal_reward[2], (self.n-1, self.n-1):goal_reward[3]}
		else:
			goal_reward = super().goal_reward

def make_env(env_name, misc_param):
	all_envs = []
	if env_name == "CartPole":
		env_params = [(9.8, 1.0, 0.1, 0.5, 10.0), (9.8, 1.0, 0.05, 0.25, 5.0), (9.8, 1.0, 0.5, 0.75, 15.0)]	
		for g, mc, mp, l, f in env_params:
			all_envs.append(TimeLimit(CartPole_CL(gravity=g, masscart=mc, masspole=mp, length=l, force=f), 200))

	elif env_name == "MountainCar":
		# env_params = [(0.002, 0.001), (0.003, 0.001), (0.001, 0.001),\
		# 				(0.002, 0.003), (0.003, 0.003), (0.001, 0.003),\
		# 				(0.002, 0.002), (0.003, 0.002), (0.001, 0.002),\
		# 				(0.002, 0.004), (0.003, 0.004), (0.001, 0.004),\
		# 				]
		env_params = [(0.002, 0.001), (0.002, 0.002), (0.003, 0.001), (0.003, 0.002)]		
		for g, f in env_params:
			all_envs.append(TimeLimit(MountainCar_CL(gravity=g, force=f), 300))

	elif env_name == "PuddleWorld":
		env_params = [[[0.35, 0.65]], [[0.65, 0.65]], [[0.65, 0.35]]]
		for p_center in env_params:
			all_envs.append(PuddleWorld_CL(p_center=p_center, puddle_width =[[0.03, 0.03]]))

	elif env_name == "ContinuousGrid":
		env_params = [[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
		#env_params = [[0, 0, 1, 1], [1, 1, 0, 0]]
		for rew in env_params:
			all_envs.append(ContinuousGrid_CL(goal_reward=rew))

	elif env_name == "DiscreteGrid":
		wiggle_list = [(0, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)] # linear env
		#wiggle_list = [(0, 0, 10, 10), (10, 10, 0, 0), (10, 0, 10, 0), (0, 10, 0, 10)]
		for wiggle in wiggle_list:
			all_envs.append(DiscreteGrid_CL(n=int(misc_param['size']), goal_reward=wiggle))

	else:
		raise NotImplementedError

	return all_envs