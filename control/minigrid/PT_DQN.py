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
parser.add_argument('--env', type=str, default="LGrid", help="Environment")
parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon in epsilon-greedy policy")
parser.add_argument('--t-episodes', type=int, default=10000, help="Total number of episodes per task")
parser.add_argument('--switch', type=int, default=1000, help="Number of episodes per task")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--lr1', type=float, default=1e-3, help="Learning rate of Permanent Network")
parser.add_argument('--lr2', type=float, default=1e-3, help="Learning rate of Transient Network")
parser.add_argument('--seed', type=int, default=0, help="Seed")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
gamma = args.gamma

def train_T_Net():
	states, actions, next_states, rewards, done, _ = exp_replay.sample()
	with torch.no_grad():
		P_next_pred, T_next_pred = Target_net(next_states)
		next_pred = (P_next_pred + T_next_pred).max(1)[0]
	P_pred, T_pred = Net(states)
	P_pred = P_pred.gather(1, actions).detach()
	T_pred = T_pred.gather(1, actions)
	targets = rewards + (1-done) * gamma * next_pred.reshape(-1, 1)
	loss = T_criterion(T_pred+P_pred, targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()

def train_P_Net():
	loss_u = 0
	u_steps = min((exp_replay.size()//args.batch_size) - 1, 100)
	for p_update in range(u_steps):
		states, actions, _, _, _, old_p_vals = exp_replay.sample()
		states = states.to(device)
		actions = actions.to(device)
		old_p_vals = old_p_vals.to(device)
		P_pred, T_pred = Net(states)
		T_pred = T_pred.gather(1, actions).detach()
		P_pred = P_pred.gather(1, actions)
		loss = P_criterion(P_pred, T_pred+old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps

def get_action(c_obs):
	c_obs = np.moveaxis(c_obs.__array__(), 3, 1)
	c_obs = torch.tensor(c_obs.reshape(-1, *c_obs.shape[2:]), dtype=torch.float).to(device)
	with torch.no_grad():
		P_vals, T_vals = Net(c_obs.unsqueeze(0))
		curr_Q_vals = P_vals + T_vals
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return P_vals[0][action], action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

envs = make_env(args.env)
for e_id in range(len(envs)):
	envs[e_id] = FrameStack(envs[e_id], 4)

filename = "PT_DQN_env_"+args.env+"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+\
	"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)+"_seed_"+str(args.seed)+"_batch_"+str(args.batch_size)

returns_seeds = np.zeros(args.t_episodes)

np.random.seed(args.seed)
random.seed(args.seed)
for e_id in range(len(envs)):
	envs[e_id].reset(seed=args.seed)
	envs[e_id].action_space.seed(args.seed)
	envs[e_id].observation_space.seed(args.seed)

Net = obj_net_two_heads(80, envs[0].action_space.n).to(device)
torch.manual_seed(args.seed)
Net.apply(weight_init)
P_opt = optim.Adam(Net.permanent_layer.parameters(), lr=args.lr1)
P_criterion = torch.nn.MSELoss()

exp_replay = expReplay_v2(batch_size=args.batch_size, device=device)

total_step_count = 0
for epi in range(args.t_episodes):

	if epi%args.switch == 0:
		if exp_replay.size() > 0:
			loss_p = train_P_Net()
			loss_P_seeds = loss_p
			exp_replay.delete()

		idx = (epi//args.switch)%len(envs)
		env = envs[idx]
		
		Net.transient_layer.apply(weight_init)
		T_opt = optim.Adam(Net.parameters(), lr=args.lr2)
		T_criterion = torch.nn.MSELoss()

		Target_net = obj_net_two_heads(80, envs[0].action_space.n).to(device)
		Target_net.load_state_dict(Net.state_dict())
	
	loss_T_epi = []

	done = False
	info = False
	c_obs = env.reset()
	step = 0
	epi_rew = 0

	while not (done or info):
		val_p, c_action = get_action(c_obs)
		n_obs, rew, done, info = env.step(c_action)
		epi_rew += gamma**step * rew

		exp_replay.store(c_obs, c_action, n_obs, rew, done, val_p.item())
		if exp_replay.size() >= args.batch_size:
			loss = train_T_Net()
			loss_T_epi.append(loss)
		
		c_obs = n_obs
		step += 1
		total_step_count += 1

		if (total_step_count+1)%200 == 0:
			Target_net.load_state_dict(Net.state_dict())

	returns_seeds[epi] = epi_rew

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_PT_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_seeds, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_seeds, f)