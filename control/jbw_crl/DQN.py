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
parser.add_argument('--env-id', type=int, default=0, help="Environment ID")
parser.add_argument('--t-steps', type=int, default=500000, help="number of steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
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

def train_Net():
	states, actions, next_states, rewards = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states)
		next_pred = next_pred.max(1)[0]
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + gamma * next_pred.reshape(-1, 1)
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()

def get_action(c_obs):
	c_obs = c_obs.__array__().flatten()
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_Q_vals = Net(c_obs.unsqueeze(0))
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_Q_vals[0][action], action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("mps")#
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

filename = "DQN_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_batch_"+str(args.batch_size)+\
		"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)+"_ftype_"+args.ftype

Net = NN_FA(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = NN_FA(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n).to(device)
Target_net.load_state_dict(Net.state_dict())

exp_replay = expReplay_NN(batch_size=args.batch_size, device=device)

env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
env.feature_space.seed(args.seed)

cs = env.reset()
loss_list = []
for step in range(args.t_steps):
	#env.render()				
	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	rewards_array[step] = rew
	exp_replay.store(cs, c_action, ns, rew)
	
	if exp_replay.size() >= args.batch_size:
		loss = train_Net()
		loss_list.append(loss)
	
	cs = ns

	if (step+1)%200 == 0:
		Target_net.load_state_dict(Net.state_dict())

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(rewards_array, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(rewards_array, f)