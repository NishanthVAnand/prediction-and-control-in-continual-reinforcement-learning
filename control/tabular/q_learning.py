import numpy as np
import pickle

from CL_envs import *
from utils import *
from features import *

from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser(description="Parameters for the code - TD on gym envs")
parser.add_argument('--t-seeds', type=int, default=30, help="Total seeds")
parser.add_argument('--env', type=str, default="DiscreteGrid-v2", help="Environment: cartpole or mountaincar")
parser.add_argument('--t-episodes', type=int, default=1000, help="number of episodes")
parser.add_argument('--switch', type=int, default=50, help="number of episodes in one environment")
parser.add_argument('--feat-type', type=str, default="RBF", help="type of features")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--reset', action="store_true")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[args.env]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])
num_split = int(misc_param['num_split'])

def agent_policy(env, state, epsilon):
	if epsilon > np.random.random():
		selected_action = env.action_space.sample()
	else:
		q_vals = []
		for act in range(env.action_space.n):
			curr_sa_feat = f_class.features(state, act)
			q_vals.append(w_1.dot(curr_sa_feat))
		q_vals_np = np.array(q_vals)
		max_actions = np.argwhere(q_vals_np == np.max(q_vals_np)).flatten()
		selected_action = np.random.choice(max_actions)
	return selected_action

def max_qval(state):
	q_vals = []
	for act in range(env.action_space.n):
		curr_sa_feat = f_class.features(state, act)
		q_vals.append(w_1.dot(curr_sa_feat))
	q_vals_np = np.array(q_vals)
	return max(q_vals_np)

all_seeds = range(args.t_seeds)
envs = make_env(args.env, misc_param)
f_class, lr1, _ = get_features_class(args.feat_type, envs[0], misc_param, args.lr1)

returns_seeds_ct = np.zeros((args.t_seeds, args.t_episodes))

for seed in all_seeds:

	for e in envs:
		set_seed(args.env, e, seed)
	w_1 = np.zeros_like(f_class.features(e.observation_space.sample(), e.action_space.sample()))

	returns_ct = np.zeros(args.t_episodes)

	for epi in range(args.t_episodes):
		if epi%args.switch == 0:
			idx = (epi//args.switch)%len(envs) # deterministic switching
			#idx = np.random.choice(np.arange(len(envs))) # stochastic switching
			if args.reset:
				w_1 = np.zeros_like(f_class.features(e.observation_space.sample(), e.action_space.sample()))
		env = envs[idx]
		cs = env.reset()
		done = False
		epi_rew = 0
		step = 0
		while not done:
			action = agent_policy(env, cs, epsilon)
			ns, rew, done, _ = env.step(action)
			epi_rew += gamma**step * rew
			curr_feature = f_class.features(cs, action)
			c_qval = w_1.dot(curr_feature)
			
			if done:
				target = rew

			else:
				n_qval = max_qval(ns)
				target = rew + gamma * n_qval

			w_1 += lr1 * (target - c_qval) * curr_feature
			cs = ns
			step += 1
		
		returns_ct[epi] = epi_rew
	
	returns_seeds_ct[seed] = returns_ct

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_seeds_ct.mean(axis=0))
	plt.title("Q-learning")
	plt.show()

ret_mean = np.mean(returns_seeds_ct, axis=0)
ret_std = np.std(returns_seeds_ct, axis=0)

filename = "Q_learning_"+str(args.reset)+"_ntasks_"+str(len(envs))+"_gamma_"+misc_param['gamma']+"_feat_"+args.feat_type+\
		"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+"_lr1_"+str(args.lr1)

if args.save:
	with open("results/"+args.env+"/"+filename+"_mean.pkl", "wb") as f:
		pickle.dump(ret_mean, f)

	with open("results/"+args.env+"/"+filename+"_std.pkl", "wb") as f:
		pickle.dump(ret_std, f)