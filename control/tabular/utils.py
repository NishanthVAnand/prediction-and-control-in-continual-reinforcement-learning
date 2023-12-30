import numpy as np
from features import *

def set_seed(env_name, env, seed):
	np.random.seed(seed)
	env.seed(seed)
	env.action_space.seed(seed)
	env.observation_space.seed(seed)

def get_features_class(feat_type, env, misc_param, lr1, lr2=None):
	n_act = env.action_space.n
	if feat_type == "gridTabular":
		n = int(misc_param['size'])
		f_class = discreteGrid_tabular(n, env.goal, n_actions=n_act)

	else:
		raise NotImplementedError

	return f_class, lr1, lr2