import numpy as np
import scipy.signal as signal
import pickle

from features import *

def basic_policy(env_name, env, state, epsilon):
	if env_name == "DiscreteGrid":
		selected_action = np.random.choice(np.arange(4), p=[0.25, 0.25, 0.25, 0.25])
		return selected_action

	elif env_name == "ContinuousGrid":
		selected_action = np.random.choice(np.arange(4))
		return selected_action

	else:
		raise NotImplementedError

def get_eval_states(env_name, num_split, env):
	if env_name == "ContinuousGrid":
		partition = np.linspace(np.array([0., 0.]), np.array([1.0, 1.0]), num_split)
		centers = np.meshgrid(*[partition[:,i] for i in range(2)])
		eval_states = np.vstack(list(zip(*[centers[i].ravel() for i in range(2)])))
		eval_states = eval_states[np.logical_and.reduce([\
			np.linalg.norm(eval_states - st, ord=1, axis=1) > env.goal_threshold for st in np.array(env.goal)\
			])]

	elif env_name == "DiscreteGrid":
		partition = np.linspace(np.array([0, 0]), np.array([env.n-1, env.n-1]), num_split)
		centers = np.meshgrid(*[partition[:,i] for i in range(2)])
		eval_states = np.vstack(list(zip(*[centers[i].ravel() for i in range(2)]))).astype('int')
	
	else:
		raise NotImplementedError

	return eval_states

def compute_return(rew_list, gamma):
	r = rew_list[::-1] # can be numpy array too
	a = [1, -gamma]
	b = [1]
	y = signal.lfilter(b, a, x=r)
	return y[::-1][0]

def set_seed(env_name, env, seed):
	np.random.seed(seed)
	env.seed(seed)
	env.action_space.seed(seed)
	env.observation_space.seed(seed)

def compute_values(f_class, w, states):
	features_states = np.apply_along_axis(f_class.features, axis=1, arr=states)
	return features_states.dot(w)

def rmsve(v_pred, v_pi, d_pi=None):
	if d_pi is not None:
		error = (d_pi.dot((np.array(v_pred) - v_pi)**2))**0.5 
	else:
		error = (((np.array(v_pred) - v_pi)**2).mean())**0.5
	return error

def get_true_vals(env_name, envs, misc_param):
	if env_name == "DiscreteGrid":
		policy = np.ones((envs[0].n ** 2, 1)) * np.array([0.25, 0.25, 0.25, 0.25])
		v_pi_list = [g.get_true_values(policy, float(misc_param['gamma'])) for g in envs]
		MC_values = np.stack(v_pi_list, axis=1)
		d_pi_list = [np.diag(env.D) for env in envs]

	else:
		file_mc = "Env_"+env_name+"_episodes_"+misc_param['episodes']+"_seeds_"+misc_param['seeds']+\
		"_num_split_"+misc_param['num_split']+"_gamma_"+misc_param['gamma']+"_epsilon_"+misc_param['epsilon']
		with open("MC_vals/"+file_mc+"_mean.pkl", "rb") as f:
			MC_values = pickle.load(f)
		d_pi_list = [None] * len(envs)
	
	return MC_values, d_pi_list

def get_features_class(feat_type, env, misc_param, lr1, lr2=None):
	if feat_type not in ["discreteFeat", "gridTabular"]:
		states_high = env.observation_space.high
		states_low = env.observation_space.low

	if feat_type == "RBF":
		order = int(misc_param['order'])
		f_class = RBF(order=order, high=states_high, low=states_low)

	elif feat_type == "discreteFeat":
		n = int(misc_param['size'])
		f_class = discreteGrid_features(n, env.goal)

	elif feat_type == "gridTabular":
		n = int(misc_param['size'])
		f_class = discreteGrid_tabular(n, env.goal)

	else:
		raise NotImplementedError

	return f_class, lr1, lr2