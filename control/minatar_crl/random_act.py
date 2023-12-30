import numpy as np
import random
import pickle
import copy

from CL_envs import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--env-name', type=str, default="breakout", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=100000, help="number of episodes")
parser.add_argument('--switch', type=int, default=5000000, help="switch env steps")
parser.add_argument('--save', action="store_true")
args = parser.parse_args()

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

returns_array = np.zeros(args.t_steps)

env = CL_envs_func(args.env_name, args.seed)
avg_return = 0
epi_return = 0
done = False
_ = env.reset()

for step in range(args.t_steps):
    
    if (step+1)%args.switch == 0:
        env = CL_envs_func(args.env_name, args.seed)
        _ = env.reset()
        epi_return = 0

    curr_action = env.action_space.sample()
    _, rew, done, _ = env.step(curr_action)
    epi_return += rew
    
    if done:
        cs = env.reset()
        avg_return = 0.99 * avg_return + 0.01 * epi_return
        epi_return = 0

    returns_array[step] = copy.copy(avg_return)


if args.save:
    filename = "Random_"+str(args.t_steps)+"_switch_"+str(args.switch)+"_env_name_"+str(args.env_name)+"_seed_"+str(args.seed)+".pkl"
    with open("results/"+filename, "wb") as f:
        pickle.dump(returns_array, f)