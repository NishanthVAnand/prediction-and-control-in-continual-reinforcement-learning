from minigrid_env import *
from gym.wrappers import *
from gym_minigrid.wrappers import *

from discrete_grid import *
from pixel_wrapper import *

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

	if env_name == "DiscreteGrid":
		wiggle_list = [(0, 0, 10, 10), (10, 10, 0, 0), (10, 0, 10, 0), (0, 10, 0, 10)] # linear env
		policy_params = [(0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25)]
		for wiggle in wiggle_list:
			all_envs.append(GridImgObservation(DiscreteGrid_CL(n=int(misc_param['size']), goal_reward=wiggle)))

	elif env_name == "ItemGrid":
		env_params = [{\
		(2,1):("blue", -1),\
		(4,1):("blue", -1),\
		(1,2):("blue", -1),\
		(3,2):("blue", -1),\
		(2,3):("blue", -1),\
		(4,3):("blue", -1),\
		(1,4):("blue", -1),\
		(3,4):("blue", -1),\
		(7,6):("purple", 1),\
		(9,6):("purple", 1),\
		(6,7):("purple", 1),\
		(8,7):("purple", 1),\
		(7,8):("purple", 1),\
		(9,8):("purple", 1),\
		(6,9):("purple", 1),\
		(8,9):("purple", 1),\
		(6,1):("yellow", 1),\
		(8,1):("yellow", 1),\
		(7,2):("yellow", 1),\
		(9,2):("yellow", 1),\
		(6,3):("yellow", 1),\
		(8,3):("yellow", 1),\
		(7,4):("yellow", 1),\
		(9,4):("yellow", 1),\
		(1,6):("red", 1),\
		(3,6):("red", 1),\
		(2,7):("red", 1),\
		(4,7):("red", 1),\
		(1,8):("red", 1),\
		(3,8):("red", 1),\
		(2,9):("red", 1),\
		(4,9):("red", 1)},\
		\
		{(2,1):("blue", 1),\
		(4,1):("blue", 1),\
		(1,2):("blue", 1),\
		(3,2):("blue", 1),\
		(2,3):("blue", 1),\
		(4,3):("blue", 1),\
		(1,4):("blue", 1),\
		(3,4):("blue", 1),\
		(7,6):("purple", 1),\
		(9,6):("purple", 1),\
		(6,7):("purple", 1),\
		(8,7):("purple", 1),\
		(7,8):("purple", 1),\
		(9,8):("purple", 1),\
		(6,9):("purple", 1),\
		(8,9):("purple", 1),\
		(6,1):("yellow", 0.5),\
		(8,1):("yellow", 0.5),\
		(7,2):("yellow", 0.5),\
		(9,2):("yellow", 0.5),\
		(6,3):("yellow", 0.5),\
		(8,3):("yellow", 0.5),\
		(7,4):("yellow", 0.5),\
		(9,4):("yellow", 0.5),\
		(1,6):("red", 1),\
		(3,6):("red", 1),\
		(2,7):("red", 1),\
		(4,7):("red", 1),\
		(1,8):("red", 1),\
		(3,8):("red", 1),\
		(2,9):("red", 1),\
		(4,9):("red", 1)},\
		\
		{(2,1):("blue", 1),\
		(4,1):("blue", 1),\
		(1,2):("blue", 1),\
		(3,2):("blue", 1),\
		(2,3):("blue", 1),\
		(4,3):("blue", 1),\
		(1,4):("blue", 1),\
		(3,4):("blue", 1),\
		(7,6):("purple", -1),\
		(9,6):("purple", -1),\
		(6,7):("purple", -1),\
		(8,7):("purple", -1),\
		(7,8):("purple", -1),\
		(9,8):("purple", -1),\
		(6,9):("purple", -1),\
		(8,9):("purple", -1),\
		(6,1):("yellow", 1),\
		(8,1):("yellow", 1),\
		(7,2):("yellow", 1),\
		(9,2):("yellow", 1),\
		(6,3):("yellow", 1),\
		(8,3):("yellow", 1),\
		(7,4):("yellow", 1),\
		(9,4):("yellow", 1),\
		(1,6):("red", 1),\
		(3,6):("red", 1),\
		(2,7):("red", 1),\
		(4,7):("red", 1),\
		(1,8):("red", 1),\
		(3,8):("red", 1),\
		(2,9):("red", 1),\
		(4,9):("red", 1)},\
		\
		{(2,1):("blue", -1),\
		(4,1):("blue", -1),\
		(1,2):("blue", -1),\
		(3,2):("blue", -1),\
		(2,3):("blue", -1),\
		(4,3):("blue", -1),\
		(1,4):("blue", -1),\
		(3,4):("blue", -1),\
		(7,6):("purple", -1),\
		(9,6):("purple", -1),\
		(6,7):("purple", -1),\
		(8,7):("purple", -1),\
		(7,8):("purple", -1),\
		(9,8):("purple", -1),\
		(6,9):("purple", -1),\
		(8,9):("purple", -1),\
		(6,1):("yellow", 0.5),\
		(8,1):("yellow", 0.5),\
		(7,2):("yellow", 0.5),\
		(9,2):("yellow", 0.5),\
		(6,3):("yellow", 0.5),\
		(8,3):("yellow", 0.5),\
		(7,4):("yellow", 0.5),\
		(9,4):("yellow", 0.5),\
		(1,6):("red", 1),\
		(3,6):("red", 1),\
		(2,7):("red", 1),\
		(4,7):("red", 1),\
		(1,8):("red", 1),\
		(3,8):("red", 1),\
		(2,9):("red", 1),\
		(4,9):("red", 1)}]

		policy_params = [[0.33, 0.33, 0.34],\
		[0.33, 0.33, 0.34],\
		[0.33, 0.33, 0.34],\
		[0.33, 0.33, 0.34]]

		for item_dict in env_params:
			all_envs.append(ImgObsWrapper(\
				OneHotPartialObsWrapper(\
				ItemEnv(item_pos_rew_dict=item_dict, rew_factor=2, vision=5))))

	else:
		raise NotImplementedError

	return all_envs, policy_params

def extractVisionField(ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		self.observation_space = self.env.observation_space["image"]

	def observation(self, obs):
		return obs["image"]

	def reset(self):
		obs = self.env.reset()
		return self.observation(obs)

	def step(self, action):
		obs, rew, terminated, truncated = self.env.step(action)
		return self.observation(obs), rew, terminated, truncated