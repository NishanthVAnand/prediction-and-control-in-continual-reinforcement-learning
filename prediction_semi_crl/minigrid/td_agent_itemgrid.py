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
parser.add_argument('--obs-type', type=str, default="obj", help="RGB or obj for vision")
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

def get_errors(states):
	with torch.no_grad():
		pred = Net(states)
		MC_curr_pred = MC_models[idx](states)
		MSE_oth = 0
		for ix in range(len(envs)):
			if ix != idx:
				MC_oth_pred = MC_models[ix](states)
				MSE_oth += rmsve(pred, MC_oth_pred).item()
	MSE_curr = rmsve(pred, MC_curr_pred).item()
	return MSE_curr, MSE_oth/(len(envs)-1)

def train():
	states, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Net(next_states)
		#MC_curr_pred = MC_models[idx](states)
		#MC_oth_pred = MC_models[1-idx](states)
	pred = Net(states)
	targets = rewards + (1-done) * gamma * next_pred
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	#MSE_curr = rmsve(pred.detach(), MC_curr_pred).item()
	#MSE_oth = rmsve(pred.detach(), MC_oth_pred).item()
	return loss.item()#, MSE_curr, MSE_oth

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
	#PATH = "models/TD_target_cnn_env_"+args.env+"_id_"+str(e_id)+"_epi_"+str(misc_param['TD_episodes'])+"_lr_"+misc_param['TD_lr']+".pt"
	MC_models[e_id].load_state_dict(torch.load(PATH, map_location=device))
	MC_models[e_id].eval()

filename = "TD_env_"+args.env+"_episodes_"+str(args.t_episodes)+"_switch_"+str(args.switch)+\
	"_lr1_"+str(args.lr1)+"_t_seeds_"+str(args.t_seeds)+"_reset_"+str(args.reset)+"_batch_"+str(args.batch_size)

loss_TD_seeds = np.zeros((args.t_seeds, args.t_episodes))
err_ct_seeds = np.zeros((args.t_seeds, args.t_episodes))
err_ot_seeds = np.zeros((args.t_seeds, args.t_episodes))

for seed in range(args.t_seeds):

	for e_id in range(len(envs)):
		set_seed(envs[e_id], seed)

	Net = obj_net(80).to(device)
	opt = optim.SGD(Net.parameters(), lr=args.lr1)
	criterion = torch.nn.MSELoss()

	exp_replay = expReplay(batch_size=args.batch_size, device=device)

	loss_list = []
	err_ct = []
	err_ot = []

	for epi in range(args.t_episodes):

		if epi%args.switch == 0:
			idx = np.random.choice(list(range(len(envs)))) #(epi//args.switch)%len(envs) #
			env = envs[idx]
			policy = policies[idx]

			if exp_replay.size() > 0:
				exp_replay.delete()

			if args.reset:
				if args.obs_type == "RGB":
					Net = cnn_net(12).to(device)
				else:
					Net = obj_net(80).to(device)
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
				#loss, MSE_c, MSE_o = train()
				loss = train()
				loss_TD_epi.append(loss)
				#err_ct_epi.append(MSE_c)
				#err_ot_epi.append(MSE_o)

			c_obs = n_obs

		if len(err_ct_epi) > 0:
			loss_TD_seeds[seed, epi] = sum(loss_TD_epi)/len(loss_TD_epi)
			#err_ct_seeds[seed, epi] = sum(err_ct_epi)/len(err_ct_epi)
			#err_ot_seeds[seed, epi] = sum(err_ot_epi)/len(err_ot_epi)

		MSE_c, MSE_o = get_errors(eval_states.to(device))
		err_ct_seeds[seed, epi] = MSE_c
		err_ot_seeds[seed, epi] = MSE_o

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