import numpy as np
import pickle
import torch
import torch.optim as optim

from gym.wrappers import *
from argparse import ArgumentParser
from configparser import ConfigParser

from CL_envs import *
from utils import *
from model import *
from replay import *

parser = ArgumentParser(description="Parameters for the code - PT Agent")
parser.add_argument('--env', type=str, default="ItemGrid", help="Environment")
parser.add_argument('--t-episodes', type=int, default=10000, help="Total number of episodes per task")
parser.add_argument('--switch', type=int, default=1000, help="Number of episodes per task")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--lr1', type=float, default=1e-3, help="Learning rate of Permanent Network")
parser.add_argument('--t-seeds', type=int, default=30, help="Total seeds")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save', action="store_true")
parser.add_argument('--save-model', action="store_true")
parser.add_argument('--reset', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[args.env]
gamma = float(misc_param['gamma'])
size = int(misc_param['size'])

def get_vals(torch_eval_states):
	with torch.no_grad():
		vals = Net(torch_eval_states.to(device))
	vals[[0, size-1, -1, -size]] = 0.0
	return vals.squeeze().cpu()

def train():
	states, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Net(next_states)
	pred = Net(states)
	targets = rewards + (1-done) * gamma * next_pred
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

envs, policies = make_env(args.env, misc_param)
eval_states = get_eval_states(args.env, envs[0], None)
MC_values = get_true_vals(args.env, envs, misc_param)
torch_eval_states = torch.stack([torch.tensor(np.moveaxis(envs[0].observation(i).__array__(), 2, 0)) for i in eval_states])

for e_id in range(len(envs)):
	envs[e_id] = FrameStack(envs[e_id], 1)

filename = "TD_env_"+args.env+"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+\
	"_lr1_"+str(args.lr1)+"_t_seeds_"+str(args.t_seeds)+"_reset_"+str(args.reset)+"_batch_"+str(args.batch_size)

loss_TD_seeds = np.zeros((args.t_seeds, args.t_episodes))
err_ct_seeds = np.zeros((args.t_seeds, args.t_episodes))
err_ot_seeds = np.zeros((args.t_seeds, args.t_episodes))

for seed in range(args.t_seeds):

	for e_id in range(len(envs)):
		set_seed(envs[e_id], seed)

	Net = cnn_net(3).to(device)	
	opt = optim.SGD(Net.parameters(), lr=args.lr1)
	criterion = torch.nn.MSELoss()

	exp_replay = expReplay(batch_size=args.batch_size, device=device)

	for epi in range(args.t_episodes):

		if epi%args.switch == 0:
			idx = (epi//args.switch)%len(envs) #np.random.choice(list(range(len(envs)))) #
			env = envs[idx]
			policy = policies[idx]

			if exp_replay.size() > 0:
				exp_replay.delete()

			if args.reset:
				Net = cnn_net(3).to(device)	
				opt = optim.SGD(Net.parameters(), lr=args.lr1)
				criterion = torch.nn.MSELoss()
		
		loss_TD_epi = []
		err_ct_epi = []
		err_ot_epi = []

		done = False
		c_obs = env.reset()

		while not done:
			n_obs, rew, done, info = env.step(basic_policy(args.env, env, policy_param=policy))
			exp_replay.store(c_obs, n_obs, rew, done)
			if exp_replay.size() >= args.batch_size:
				loss = train()
				loss_TD_epi.append(loss)

			c_obs = n_obs

		if len(err_ct_epi) > 0:
			loss_TD_seeds[seed, epi] = sum(loss_TD_epi)/len(loss_TD_epi)

		vals = get_vals(torch_eval_states)
		err_ct_seeds[seed, epi] = rmsve(vals, MC_values[:, idx], d_pi=None)
		err_ot_seeds[seed, epi] = np.array([rmsve(vals, MC_values[:, ix], d_pi=None)\
			for ix in range(len(envs)) if ix != idx]).mean()

	if args.save_model:
		torch.save(Net.state_dict(), "models/"+filename+"_TDNet_seed_"+str(seed)+".pt")

err_ct_mean = np.mean(err_ct_seeds, axis=0)
err_ct_std = np.std(err_ct_seeds, axis=0)
err_ot_mean = np.mean(err_ot_seeds, axis=0)
err_ot_std = np.std(err_ot_seeds, axis=0)

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(err_ct_mean, 'b')
	plt.plot(err_ot_mean, 'r')
	plt.show()
	#print(err_ct)

if args.save:
	with open("results/"+filename+"_curr_errors_mean.pkl", "wb") as f:
		pickle.dump(err_ct_mean, f)

	with open("results/"+filename+"_oth_errors_mean.pkl", "wb") as f:
		pickle.dump(err_ot_mean, f)

	with open("results/"+filename+"_curr_errors_std.pkl", "wb") as f:
		pickle.dump(err_ct_std, f)

	with open("results/"+filename+"_oth_errors_std.pkl", "wb") as f:
		pickle.dump(err_ot_std, f)