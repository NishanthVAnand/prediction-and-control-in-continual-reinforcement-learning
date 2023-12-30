import numpy as np
import pickle

from CL_envs import *
from utils import *
from features import *

from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser(description="Parameters for the code - TD on gym envs")
parser.add_argument('--t-seeds', type=int, default=30, help="Total seeds")
parser.add_argument('--env', type=str, default="MountainCar", help="Environment: cartpole or mountaincar")
parser.add_argument('--t-episodes', type=int, default=1000, help="number of episodes")
parser.add_argument('--switch', type=int, default=50, help="number of episodes in one environment")
parser.add_argument('--feat-type', type=str, default="RBF", help="type of features")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--save', action="store_true")
parser.add_argument('--reset', action="store_true")
parser.add_argument('--plot', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[args.env]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])
num_split = int(misc_param['num_split'])

all_seeds = range(args.t_seeds)
envs = make_env(args.env, misc_param)
eval_states = get_eval_states(args.env, num_split, envs[0])
MC_values, d_pi_list = get_true_vals(args.env, envs, misc_param)
f_class, lr1, _ = get_features_class(args.feat_type, envs[0], misc_param, args.lr1)

err_seeds_ct = np.zeros((args.t_seeds, args.t_episodes)) # current task errors
err_seeds_ot = np.zeros((args.t_seeds, args.t_episodes)) # other task errors
td_values = np.zeros((args.t_seeds, args.t_episodes//args.switch, eval_states.shape[0]))

for seed in all_seeds:

	for e in envs:
		set_seed(args.env, e, seed)
	w_1 = np.zeros_like(f_class.features(e.observation_space.sample()))
	err_ct = np.zeros(args.t_episodes)
	err_ot = np.zeros(args.t_episodes)

	for epi in range(args.t_episodes):
		
		if epi%args.switch == 0:
			idx = (epi//args.switch)%len(envs)
			#idx = np.random.choice(np.arange(len(envs)))
			if args.reset:
				w_1 = np.zeros_like(f_class.features(e.observation_space.sample()))
		env = envs[idx]
		cs = env.reset()
		done = False

		while not done:
			action = basic_policy(args.env, env, cs, epsilon)
			ns, rew, done, _ = env.step(action)
			curr_feature = f_class.features(cs)
			cs_val_p = w_1.dot(curr_feature)
			
			if done:
				target = rew

			else:
				ns_val_p = w_1.dot(f_class.features(ns))
				target = rew + gamma * ns_val_p

			w_1 += lr1 * (target - cs_val_p) * curr_feature
			cs = ns

		value_estimates = compute_values(f_class, w_1, eval_states)
		
		err_ct[epi] = rmsve(value_estimates, MC_values[:, idx], d_pi=d_pi_list[idx])
		err_ot[epi] = np.array([rmsve(value_estimates, MC_values[:, ix], d_pi=None)\
			for ix in range(len(envs)) if ix != idx]).mean()

		td_values[seed, epi//args.switch, :] = value_estimates

	err_seeds_ct[seed] = err_ct
	err_seeds_ot[seed] = err_ot

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(err_seeds_ct.mean(axis=0))
	plt.plot(err_seeds_ot.mean(axis=0))
	plt.show()

filename = "TD"+"_reset_"+str(args.reset)+"_ntasks_"+str(len(envs))+"_gamma_"+misc_param['gamma']+"_feat_"+args.feat_type+\
		"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+"_lr1_"+str(args.lr1)

err_mean = np.mean(err_seeds_ct, axis=0)
err_std = np.std(err_seeds_ct, axis=0)

err_mean_ot = np.mean(err_seeds_ot, axis=0)
err_std_ot = np.std(err_seeds_ot, axis=0)

if args.save:
	with open("results/"+args.env+"/"+filename+"_mean.pkl", "wb") as f:
		pickle.dump(err_mean, f)

	with open("results/"+args.env+"/"+filename+"_std.pkl", "wb") as f:
		pickle.dump(err_std, f)

	with open("results/"+args.env+"/"+filename+"_ot_mean.pkl", "wb") as f:
		pickle.dump(err_mean_ot, f)

	with open("results/"+args.env+"/"+filename+"_ot_std.pkl", "wb") as f:
		pickle.dump(err_std_ot, f)

	with open("results/"+args.env+"/"+filename+"_td_values.pkl", "wb") as f:
		pickle.dump(td_values, f)