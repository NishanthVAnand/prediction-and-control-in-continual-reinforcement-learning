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
parser.add_argument('--obs-type', type=str, default="obj", help="RGB or obj for vision")
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

def get_errors(states):
	with torch.no_grad():
		P_pred, T_pred = Net(states)
		MC_curr_pred = MC_models[idx](states)
		MSE_oth = 0
		for ix in range(len(envs)):
			if ix != idx:
				MC_oth_pred = MC_models[ix](states)
				MSE_oth += rmsve(P_pred, MC_oth_pred).item()
	MSE_curr = rmsve(T_pred+P_pred, MC_curr_pred).item()
	return MSE_curr, MSE_oth/(len(envs)-1)

def train_T_Net():
	states, next_states, rewards, done, _ = exp_replay.sample()
	with torch.no_grad():
		P_next_pred, T_next_pred = Net(next_states)
		P_pred, _ = Net(states)
	_, T_pred = Net(states)
	targets = rewards + (1-done) * gamma * (P_next_pred + T_next_pred)
	loss = T_criterion(T_pred+P_pred, targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()

def train_P_Net():
	loss_u = 0
	u_steps = (exp_replay.size()//64) - 1
	for p_update in range(u_steps):
		curr_batch = list(itertools.islice(exp_replay.memory, p_update*64, (p_update+1)*64))
		states, _, _, _, old_p_vals = map(torch.stack, zip(*curr_batch))
		states = states.to(device)
		old_p_vals = old_p_vals.to(device)
		with torch.no_grad():
			_, T_pred = Net(states)
		P_pred, _ = Net(states)
		loss = P_criterion(P_pred, T_pred+old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

envs, policies = make_env(args.env, misc_param)
eval_states = torch.tensor([])
for e_id in range(len(envs)):
	envs[e_id] = FrameStack(envs[e_id], 4)
	set_seed(envs[e_id], 100+e_id)
	eval_states = torch.cat((eval_states, get_eval_states(args.env, envs[e_id], policies[e_id])))

MC_models = []
for e_id in range(len(envs)):
	MC_models.append(obj_net(80).to(device))
	PATH = "models/MC_returns_cnn_env_"+args.env+"_id_"+str(e_id)+"_epi_"+misc_param['MC_episodes']+"_lr_"+misc_param['MC_lr']+".pt"
	MC_models[e_id].load_state_dict(torch.load(PATH, map_location=device))
	MC_models[e_id].eval()

filename = "PT_env_"+args.env+"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+\
	"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)+"_t_seeds_"+str(args.t_seeds)+"_batch_"+str(args.batch_size)

loss_T_seeds = np.zeros((args.t_seeds, args.t_episodes))
loss_P_seeds = np.zeros((args.t_seeds, args.t_episodes//args.switch))
err_ct_seeds = np.zeros((args.t_seeds, args.t_episodes))
err_ot_seeds = np.zeros((args.t_seeds, args.t_episodes))

for seed in range(args.t_seeds):

	for e_id in range(len(envs)):
		set_seed(envs[e_id], seed)

	Net = obj_net_two_heads(80).to(device)
	P_opt = optim.SGD(Net.permanent_layer.parameters(), lr=args.lr1)
	P_criterion = torch.nn.MSELoss()

	exp_replay = expReplay_v2(batch_size=args.batch_size, device=device)

	for epi in range(args.t_episodes):

		if epi%args.switch == 0:
			if exp_replay.size() > 0:
				loss_p = train_P_Net()
				loss_P_seeds = loss_p
				exp_replay.delete()
			idx = np.random.choice(list(range(len(envs)))) #(epi//args.switch)%len(envs)#
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

		MSE_c, MSE_o = get_errors(eval_states.to(device))
		err_ct_seeds[seed, epi] = MSE_c
		err_ot_seeds[seed, epi] = MSE_o

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