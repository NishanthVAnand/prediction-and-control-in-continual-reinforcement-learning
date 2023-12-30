import torch
import numpy as np
import scipy.signal as signal
import pickle

from CL_envs import *

def get_task(env_name, misc_param):
	tasks, policies = make_env(env_name, misc_param)
	return tasks, policies

def basic_policy(env_name, env, **kwargs):
	if env_name in ["ItemGrid"]:
		selected_action = np.random.choice([0, 1, 2], p=kwargs['policy_param'])

	elif env_name == "DiscreteGrid":
		selected_action = np.random.choice([0, 1, 2, 3], p=kwargs['policy_param'])		

	else:
		raise NotImplementedError
	
	return selected_action

def compute_return(rew_list, gamma):
	r = rew_list[::-1] # can be numpy array too
	a = [1, -gamma]
	b = [1]
	y = signal.lfilter(b, a, x=r)
	return y[::-1]

def set_seed(env, seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	env.reset(seed=seed)
	env.action_space.seed(seed)
	env.observation_space.seed(seed)

def rmsve(v_pred, v_pi, d_pi=None):
	if d_pi is not None:
		error = (d_pi.dot((v_pred - v_pi)**2))**0.5 
	else:
		error = (((v_pred - v_pi)**2).mean())**0.5
	return error

def msve(v_pred, v_pi, d_pi=None):
	if d_pi is not None:
		error = (d_pi.dot((v_pred - v_pi)**2))
	else:
		error = (((v_pred - v_pi)**2).mean())
	return error

def get_eval_states(env_name, env, policy):
	if env_name in ["ItemGrid"]:
		eval_states = []
		for i in range(15):
			cs = env.reset()
			cs = np.moveaxis(cs.__array__(), 3, 1)
			eval_states.append(torch.tensor(cs.reshape(-1, *cs.shape[2:]), dtype=torch.float))
			done = False
			while not done:	
				cs, _, done, _ = env.step(basic_policy(env_name	, env, policy_param=policy))
				cs = np.moveaxis(cs.__array__(), 3, 1)
				eval_states.append(torch.tensor(cs.reshape(-1, *cs.shape[2:]), dtype=torch.float))
		return torch.stack(eval_states)

	elif env_name == "DiscreteGrid":
		partition = np.linspace(np.array([0, 0]), np.array([env.n-1, env.n-1]), 5)
		centers = np.meshgrid(*[partition[:,i] for i in range(2)])
		eval_states = np.vstack(list(zip(*[centers[i].ravel() for i in range(2)]))).astype('int')
		return eval_states
	
	else:
		raise NotImplementedError

	return None

def get_true_vals(env_name, envs, misc_param):
	if env_name == "DiscreteGrid":
		policy = np.ones((envs[0].n ** 2, 1)) * np.array([0.25, 0.25, 0.25, 0.25])
		v_pi_list = [g.get_true_values(policy, float(misc_param['gamma'])) for g in envs]
		MC_values = np.stack(v_pi_list, axis=1)
	return MC_values