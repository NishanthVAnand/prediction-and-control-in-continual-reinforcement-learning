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
parser.add_argument('--seed', type=int, default=0, help="Seed")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save', action="store_true")
parser.add_argument('--reset', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
gamma = args.gamma

def train_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states)
		next_pred = next_pred.max(1)[0]
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + (1-done) * gamma * next_pred.reshape(-1, 1)
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()

def get_action(c_obs):
	c_obs = np.moveaxis(c_obs.__array__(), 3, 1)
	c_obs = torch.tensor(c_obs.reshape(-1, *c_obs.shape[2:]), dtype=torch.float).to(device)
	with torch.no_grad():
		curr_Q_vals = Net(c_obs.unsqueeze(0))
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_Q_vals[0][action], action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

envs = make_env(args.env)
for e_id in range(len(envs)):
	envs[e_id] = FrameStack(envs[e_id], 4)

filename = "DQN_env_"+args.env+"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+\
	"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)+"_batch_"+str(args.batch_size)+"_reset_"+str(args.reset)

returns_seeds = np.zeros(args.t_episodes)

np.random.seed(args.seed)
random.seed(args.seed)
for e_id in range(len(envs)):
	envs[e_id].reset(seed=args.seed)
	envs[e_id].action_space.seed(args.seed)
	envs[e_id].observation_space.seed(args.seed)

Net = obj_net(80, envs[0].action_space.n).to(device)
torch.manual_seed(args.seed)
Net.apply(weight_init)
Net.critic.apply(weight_init)
Net.critic.apply(weight_init)

if not args.reset:
	Net = obj_net(80, envs[0].action_space.n).to(device) # reinitializing the network for reproducibility across algos
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

exp_replay = expReplay(batch_size=args.batch_size, device=device)

total_step_count = 0
for epi in range(args.t_episodes):

	if epi%args.switch == 0:
		exp_replay.delete()
		idx = (epi//args.switch)%len(envs)
		env = envs[idx]
		
		if args.reset:
			Net = obj_net(80, envs[0].action_space.n).to(device)
			opt = optim.Adam(Net.parameters(), lr=args.lr1)
			criterion = torch.nn.MSELoss()

		Target_net = obj_net(80, envs[0].action_space.n).to(device)
		Target_net.load_state_dict(Net.state_dict())				
	
	loss_T_epi = []

	done = False
	info = False
	c_obs = env.reset()
	step = 0
	epi_rew = 0

	while not (done or info):
		_, c_action = get_action(c_obs)
		n_obs, rew, done, info = env.step(c_action)
		epi_rew += gamma**step * rew

		exp_replay.store(c_obs, c_action, n_obs, rew, done)
		if exp_replay.size() >= args.batch_size:
			loss = train_Net()
			loss_T_epi.append(loss)
		
		c_obs = n_obs
		step += 1
		total_step_count += 1

		if (total_step_count+1)%200 == 0:
			Target_net.load_state_dict(Net.state_dict())

	returns_seeds[epi] = epi_rew

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_seeds, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_seeds, f)