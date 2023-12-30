import gym
from empty_grid_env import *
from gym.wrappers import *
from gym_minigrid.wrappers import *

def make_env(env_name):
	all_envs = []
	if env_name == "LGrid":
		env_params = [{(7, 1):5, (9, 3):0}, {(7, 1):0, (9, 3):5}]
		for e in env_params:
			all_envs.append(ImgObsWrapper(OneHotPartialObsWrapper(LGrid(goal_pos_rew=e))))

	else:
		raise NotImplementedError

	return all_envs