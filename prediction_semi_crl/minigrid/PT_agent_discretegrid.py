import numpy as np
import pickle
import itertools
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
parser.add_argument('--lr2', type=float, default=1e-3, help="Learning rate of Transient Network")
parser.add_argument('--t-seeds', type=int, default=30, help="Total seeds")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[args.env]
gamma = float(misc_param['gamma'])
size = int(misc_param['size'])

def get_vals(torch_eval_states):
	with torch.no_grad():
		vals_p, vals_t = Net(torch_eval_states.to(device))
	vals_p[[0, size-1, -1, -size]] = 0.0
	vals_t[[0, size-1, -1, -size]] = 0.0
	return vals_p.squeeze().cpu(), vals_t.squeeze().cpu()

def train_T_Net():
	states, next_states, rewards, done, _ = exp_replay.sample()
	with torch.no_grad():
		P_next_pred, T_next_pred = Net(next_states)
	P_pred, T_pred = Net(states)
	targets = rewards + (1-done) * gamma * (P_next_pred + T_next_pred)
	loss = T_criterion(T_pred+P_pred.detach(), targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()

def train_P_Net():
	loss_u = 0
	u_steps = 100
	for p_update in range(u_steps):
		states, _, _, _, old_p_vals = exp_replay.sample()
		states = states.to(device)
		old_p_vals = old_p_vals.to(device)
		P_pred, T_pred = Net(states)
		loss = P_criterion(P_pred, T_pred.detach()+old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

envs, policies = make_env(args.env, misc_param)
eval_states = get_eval_states(args.env, envs[0], None)
MC_values = get_true_vals(args.env, envs, misc_param)
torch_eval_states = torch.stack([torch.tensor(np.moveaxis(envs[0].observation(i).__array__(), 2, 0)) for i in eval_states])

for e_id in range(len(envs)):
	envs[e_id] = FrameStack(envs[e_id], 1)
filename = "PT_env_"+args.env+"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+\
	"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)+"_t_seeds_"+str(args.t_seeds)+"_batch_"+str(args.batch_size)

loss_T_seeds = np.zeros((args.t_seeds, args.t_episodes))
loss_P_seeds = np.zeros((args.t_seeds, args.t_episodes//args.switch))
err_ct_seeds = np.zeros((args.t_seeds, args.t_episodes))
err_ot_seeds = np.zeros((args.t_seeds, args.t_episodes))

for seed in range(args.t_seeds):

	for e_id in range(len(envs)):
		set_seed(envs[e_id], seed)

	Net = cnn_net_two_heads(3).to(device)
	P_opt = optim.SGD(Net.permanent_layer.parameters(), lr=args.lr1)
	P_criterion = torch.nn.MSELoss()

	exp_replay = expReplay_v2(batch_size=args.batch_size, device=device)

	for epi in range(args.t_episodes):

		if epi%args.switch == 0:
			if exp_replay.size() > 0:
				loss_p = train_P_Net()
				loss_P_seeds = loss_p
				exp_replay.delete()
			idx = (epi//args.switch)%len(envs) #np.random.choice(list(range(len(envs)))) #
			env = envs[idx]
			policy = policies[idx]
			
			Net.transient_layer.apply(weight_init)
			T_opt = optim.SGD(Net.parameters(), lr=args.lr2)
			T_criterion = torch.nn.MSELoss()
		
		loss_T_epi = []
		err_ct_epi = []
		err_ot_epi = []
		data = []

		done = False
		c_obs = env.reset()

		while not done:
			n_obs, rew, done, info = env.step(basic_policy(args.env, env, policy_param=policy))
			with torch.no_grad():
				val_p, _ = Net(torch.tensor(c_obs.__array__().reshape(-1, *c_obs.shape[1:3]), dtype=torch.float).unsqueeze(0).to(device))
			exp_replay.store(c_obs, n_obs, rew, done, val_p.item())
			if exp_replay.size() >= args.batch_size:
				loss = train_T_Net()
				loss_T_epi.append(loss)

			c_obs = n_obs

		if len(err_ct_epi) > 0:
			loss_T_seeds[seed, epi] = sum(loss_T_epi)/len(loss_T_epi)

		p_vals, t_vals = get_vals(torch_eval_states)
		err_ct_seeds[seed, epi] = rmsve(p_vals+t_vals, MC_values[:, idx], d_pi=None)
		err_ot_seeds[seed, epi] = np.array([rmsve(p_vals, MC_values[:, ix], d_pi=None)\
			for ix in range(len(envs)) if ix != idx]).mean()

	if args.save_model:
		torch.save(Net.state_dict(), "models/"+filename+"_PT_Net_seed_"+str(seed)+".pt")

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