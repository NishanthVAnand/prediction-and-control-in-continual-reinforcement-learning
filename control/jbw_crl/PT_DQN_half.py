import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim

from wrappers import *
from model import *
from replay import *
from env_params import *

from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--t-steps', type=int, default=500000, help="number of steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--lr2', type=float, default=0.1, help="learning rate for transient values")
parser.add_argument('--update', type=int, default=10000, help="PM update frequency")
parser.add_argument('--decay', type=float, default=0, help="decay transient weights after transfer")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--ftype', type=str, default="obj", help="Type of features")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config['JBW']
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])

def train_T_Net():
	states, actions, next_states, rewards = exp_replay.sample()
	with torch.no_grad():
		T_next_pred = Target_net(next_states)
		P_next_pred = P_Net(next_states)
		P_pred = P_Net(states)
		P_pred = P_pred.gather(1, actions)
	T_pred = T_Net(states)
	T_pred = T_pred.gather(1, actions)
	targets = rewards + gamma * ((P_next_pred + T_next_pred).max(1)[0]).reshape(-1, 1)
	loss = T_criterion(T_pred+P_pred, targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()

def train_P_Net():
	loss_u = 0
	u_steps = (exp_replay_PM.size()//args.batch_size) - 1
	for p_update in range(u_steps):
		curr_batch = list(itertools.islice(exp_replay_PM.memory, p_update*args.batch_size, (p_update+1)*args.batch_size))
		states, actions, old_p_vals = map(torch.stack, zip(*curr_batch))
		states = states.to(device)
		actions = actions.to(device)
		old_p_vals = old_p_vals.to(device)
		with torch.no_grad():
			T_pred = T_Net(states).gather(1, actions)
		P_pred = P_Net(states).gather(1, actions)
		loss = P_criterion(P_pred, T_pred+old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps

def get_action(c_obs):
	c_obs = c_obs.__array__().flatten()
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_T_vals = T_Net(c_obs.unsqueeze(0))
		curr_P_vals = P_Net(c_obs.unsqueeze(0))
		curr_Q_vals = curr_T_vals + curr_P_vals
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_P_vals[0][action], action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("mps")#
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

sim_config = make_config()
reward_fn = make_reward()
env = JBWEnv(sim_config, reward_fn, render=False, f_type=args.ftype)
env = FrameStack(env, 4)

rewards_array = np.zeros(args.t_steps)

filename = "PT_DQN_0.5x"+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_update_"+str(args.update)+"_decay_"+str(args.decay)+\
		"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)+"_batch_"+str(args.batch_size)+"_seed_"+str(args.seed)+\
		"_ftype_"+args.ftype

T_Net = NN_FA_half(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n).to(device)
T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)
T_criterion = torch.nn.MSELoss()

P_Net = NN_FA_half(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n).to(device)
P_opt = optim.SGD(P_Net.parameters(), lr=args.lr1)
P_criterion = torch.nn.MSELoss()

Target_net = NN_FA_half(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n).to(device)
Target_net.load_state_dict(T_Net.state_dict())

exp_replay = expReplay_NN(batch_size=args.batch_size, device=device)
exp_replay_PM = expReplay_NN_PM(batch_size=args.batch_size, device=device)

env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
env.feature_space.seed(args.seed)

cs = env.reset()
loss_T_list = []
loss_P_list = []
for step in range(args.t_steps):
	#env.render()				
	val_p, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	rewards_array[step] = rew
	exp_replay.store(cs, c_action, ns, rew)
	exp_replay_PM.store(cs, c_action, val_p)
	cs = ns

	if exp_replay.size() >= args.batch_size:
		loss = train_T_Net()
		loss_T_list.append(loss)

	if (step+1)%args.update == 0:
		p_loss = train_P_Net()
		loss_P_list.append(p_loss)
		for params in T_Net.parameters():
			params.data *= args.decay

	if (step+1)%200 == 0:
		Target_net.load_state_dict(T_Net.state_dict())

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(rewards_array, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(rewards_array, f)