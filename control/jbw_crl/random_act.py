import numpy as np
from argparse import ArgumentParser
import pickle
import gym

from env_params import *

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--t-steps', type=int, default=100000, help="number of episodes")
parser.add_argument('--save', action="store_true")
args = parser.parse_args()

sim_config = make_config()
reward_fn = make_reward()
env = JBWEnv(sim_config, reward_fn, True)

np.random.seed(args.seed)
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
env.feature_space.seed(args.seed)

_ = env.reset()
rewards_array = np.zeros(args.t_steps)
for step in range(args.t_steps):
    curr_action = env.action_space.sample()
    _, rew, _, _ = env.step(curr_action)
    rewards_array[step] = rew

if args.save:
    filename = "Random_"+str(args.t_steps)+"_seed_"+str(args.seed)+".pkl"
    with open("results/"+filename, "wb") as f:
        pickle.dump(rewards_array, f)