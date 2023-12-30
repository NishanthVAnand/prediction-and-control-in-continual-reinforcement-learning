import numpy as np
import pickle

from CL_envs import *
from utils import *

from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument('--t-seeds', type=int, default=30, help="Total seeds")
parser.add_argument('--env', type=str, default="DiscreteGrid", help="Environment: cartpole or mountaincar")
parser.add_argument('--t-episodes', type=int, default=1000, help="number of episodes")
parser.add_argument('--switch', type=int, default=50, help="number of episodes in one environment")
parser.add_argument('--feat-type', type=str, default="RBF", help="type of features")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--lr2', type=float, default=0.1, help="learning rate for transient values")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[args.env]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])
num_split = int(misc_param['num_split'])

def update_w1(w_1, replay):
	w_1_copy = w_1.copy()
	for c_f_1, c_f_2 in replay:
		w_1 += lr1 * (w_1_copy.dot(c_f_1) + w_2.dot(c_f_2) - w_1.dot(c_f_1)) * c_f_1
	return w_1

all_seeds = range(args.t_seeds)
envs = make_env(args.env, misc_param)
eval_states = get_eval_states(args.env, num_split, envs[0])

MC_values, d_pi_list = get_true_vals(args.env, envs, misc_param)
f_class, lr1, lr2 = get_features_class(args.feat_type, envs[0], misc_param, args.lr1, args.lr2)

err_seeds_ct = np.zeros((args.t_seeds, args.t_episodes)) # current task errors
err_seeds_ot = np.zeros((args.t_seeds, args.t_episodes)) # other task errors
avg_values = np.zeros((args.t_seeds, args.t_episodes//args.switch, eval_states.shape[0]))
res_values = np.zeros((args.t_seeds, args.t_episodes//args.switch, eval_states.shape[0]))

for seed in all_seeds:

	for e in envs:
		set_seed(args.env, e, seed)
	w_1 = np.zeros_like(f_class.features(e.observation_space.sample()))
	w_2 = np.zeros_like(w_1)
	replay = []
	err_ct = np.zeros(args.t_episodes)
	err_ot = np.zeros(args.t_episodes)
	for epi in range(args.t_episodes):
		
		if epi%args.switch == 0:
			if len(replay) > 0:
				w_1 = update_w1(w_1, replay)
				del replay[:]
			idx = (epi//args.switch)%len(envs)
			#idx = np.random.choice(np.arange(len(envs)))
			w_2 = np.zeros_like(w_1)
		env = envs[idx]
		cs = env.reset()
		done = False

		while not done:
			action = basic_policy(args.env, env, cs, epsilon)
			ns, rew, done, _ = env.step(action)
			
			curr_feature = f_class.features(cs)
			cs_val_p = w_1.dot(curr_feature)
			cs_val_t = w_2.dot(curr_feature)

			replay.append([curr_feature, curr_feature])
			
			if done:
				target_2 = rew

			else:
				next_feature = f_class.features(ns)
				ns_val_p = w_1.dot(next_feature)
				ns_val_t = w_2.dot(next_feature)
				target_2 = rew + gamma * (ns_val_p + ns_val_t)

			w_2 += lr2 * (target_2 - cs_val_p - cs_val_t) * curr_feature
			cs = ns

		average_values = compute_values(f_class, w_1, eval_states)
		residual_values = compute_values(f_class, w_2, eval_states)
		value_estimates = average_values + residual_values

		err_ct[epi] = rmsve(value_estimates, MC_values[:, idx], d_pi=d_pi_list[idx])
		err_ot[epi] = np.array([rmsve(average_values, MC_values[:, ix], d_pi=None)\
			for ix in range(len(envs)) if ix != idx]).mean()

		avg_values[seed, epi//args.switch, :] = average_values
		res_values[seed, epi//args.switch, :] = residual_values

	err_seeds_ct[seed] = err_ct
	err_seeds_ot[seed] = err_ot

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(err_seeds_ct.mean(axis=0))
	plt.plot(err_seeds_ot.mean(axis=0))
	plt.show()

filename = "PT_Mem"+"_ntasks_"+str(len(envs))+"_gamma_"+misc_param['gamma']+"_feat_"+args.feat_type+\
		"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)

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

	with open("results/"+args.env+"/"+filename+"_avg_values.pkl", "wb") as f:
		pickle.dump(avg_values, f)

	with open("results/"+args.env+"/"+filename+"_res_values.pkl", "wb") as f:
		pickle.dump(res_values, f)