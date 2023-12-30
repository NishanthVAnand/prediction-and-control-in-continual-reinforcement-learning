import numpy as np
import pickle
import multiprocessing

from CL_envs import *
from utils import *

from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser(description="Parameters for the code - MC values")
parser.add_argument('--n-workers', type=int, default=8, help="Total seeds")
parser.add_argument('--env', type=str, default="MountainCar", help="Environment: CP, MC, PW")
parser.add_argument('--save', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[args.env]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])
num_split = int(misc_param['num_split'])
t_episodes = int(misc_param['episodes'])
t_seeds = int(misc_param['seeds'])

def main(seed_id: int, state_id: int, env_id: int):
	seed = all_seeds[seed_id]
	state = eval_states[state_id]
	env = envs[env_id]
	
	set_seed(args.env, env, seed)

	epi_mc_val = []
	for episode in range(t_episodes):
		rew_list = []
		cs = env.reset(state=state.copy())
		done = False
		while not done:
			action = basic_policy(args.env, env, cs, epsilon)
			ns, rew, done, _ = env.step(action)
			rew_list.append(rew)
			cs = ns
		mc_ret = compute_return(rew_list, gamma)
		epi_mc_val.append(mc_ret)
	return seed_id, state_id, env_id, np.array(epi_mc_val).mean()

all_seeds = range(t_seeds)
envs = make_env(args.env, misc_param)
eval_states = get_eval_states(args.env, num_split, envs[0])

if __name__ == "__main__":
	all_value_est = np.ndarray((t_seeds, len(eval_states), len(envs)), dtype=np.float32)
	all_value_est[:] = 0

	with multiprocessing.Pool(processes=args.n_workers) as pool:
		for seed, sid, eid, val in pool.starmap(main, [
			(seed, state_id, env_id) for seed in all_seeds
			for env_id in range(len(envs))
			for state_id in range(len(eval_states))
			]):

			all_value_est[seed, sid, eid] = val

	print(all_value_est.mean(axis=0))

	if args.save:
		filename = "Env_"+str(args.env)+"_episodes_"+str(t_episodes)+"_seeds_"+str(t_seeds)+\
		"_num_split_"+str(num_split)+"_gamma_"+str(gamma)+"_epsilon_"+str(epsilon)

		with open("MC_vals/"+filename+"_mean.pkl", "wb") as f:
			pickle.dump(all_value_est.mean(axis=0), f)

		with open("MC_vals/"+filename+"_std.pkl", "wb") as f:
			pickle.dump(all_value_est.std(axis=0), f)