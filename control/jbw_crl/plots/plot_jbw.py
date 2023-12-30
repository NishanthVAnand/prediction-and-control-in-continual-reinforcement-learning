import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.seterr(invalid='ignore')
plt.style.use('seaborn-v0_8-white')

def moving_average(a, n=3):
    cumsum_vec = np.cumsum(np.insert(a, 0, 0)) 
    ma_vec = (cumsum_vec[n:] - cumsum_vec[:-n]) / n
    return np.concatenate((a[0:n-1]/n, ma_vec))

seeds = list(range(30))
t_steps = 2100000
gamma = 0.9
update = 10000
switch = 150000
t_seeds = len(seeds)

smooth_coeff = 10000
z_star = 1.3

pdf = matplotlib.backends.backend_pdf.PdfPages("plots/JBW_performance_PT_mem_half.pdf")
fig, ax = plt.subplots(figsize=(12,5))
ax.set_rasterized(True)

### PT-DQN ###
best_dec = 0.75
best_lr1 = 1e-7
best_lr2 = 1e-3
seeds_returns = np.zeros((t_seeds, t_steps))
for s in seeds:
    fname = "PT_DQN_0.5x"+"_gamma_"+str(gamma)+\
        "_steps_"+str(t_steps)+"_update_"+str(update)+"_decay_"+str(best_dec)+\
        "_lr1_"+str(best_lr1)+"_lr2_"+str(best_lr2)+"_batch_"+str(64)+"_seed_"+str(s)+\
        "_ftype_obj"
    with open("results/"+fname+"_returns.pkl", "rb") as f:
        seeds_returns[s] = moving_average(pickle.load(f), n=smooth_coeff)
rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="PT-DQN-half", lw=1.0, color="blue", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="blue")

### DQN ###
best_lr_dqn = 1e-4
seeds_returns = np.zeros((t_seeds, t_steps))
for s in seeds:
    fname = "DQN"+"_gamma_"+str(gamma)+\
            "_steps_"+str(t_steps)+"_batch_"+str(64)+\
            "_lr1_"+str(best_lr_dqn)+"_seed_"+str(s)+"_ftype_obj"
    with open("results/"+fname+"_returns.pkl", "rb") as f:
        seeds_returns[s] = moving_average(pickle.load(f), n=smooth_coeff)
rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN", lw=1.0, color="green", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="green")

### DQN multi task ###
best_lr_dqn_mt = 1e-4
seeds_returns = np.zeros((t_seeds, t_steps))
for s in seeds:
    fname = "DQN_multi_task"+"_gamma_"+str(gamma)+\
            "_steps_"+str(t_steps)+"_batch_"+str(64)+\
            "_lr1_"+str(best_lr_dqn_mt)+"_seed_"+str(s)+"_ftype_obj"
    with open("results/"+fname+"_returns.pkl", "rb") as f:
        seeds_returns[s] = moving_average(pickle.load(f), n=smooth_coeff)
rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN (multi-task)", lw=1.0, color="brown", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="brown")

### DQN large buffer ###
best_lrs_dqn_large = 1e-4
seeds_returns = np.zeros((t_seeds, t_steps))
for s in seeds:
    fname = "DQN_large_buffer"+"_gamma_"+str(gamma)+\
            "_steps_"+str(t_steps)+"_batch_"+str(64)+\
            "_lr1_"+str(best_lrs_dqn_large)+"_seed_"+str(s)+"_ftype_obj"
    with open("results/"+fname+"_returns.pkl", "rb") as f:
        seeds_returns[s] = moving_average(pickle.load(f), n=smooth_coeff)
rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN (large buffer)", lw=1.0, color="black", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="black")

### Random ###
seeds_returns = np.zeros((t_seeds, t_steps))
for s in seeds:
    fname = "Random_"+str(t_steps)+"_seed_"+str(s)
    with open("results/"+fname+".pkl", "rb") as f:
        seeds_returns[s] = moving_average(pickle.load(f), n=smooth_coeff)
rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="Random", lw=1.0, color="red", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="red")
    
for v_cord in np.arange(switch-1, t_steps, switch):
    plt.axvline(x=v_cord, color='k', ls=':', alpha=0.8, lw=0.6)
custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='brown', lw=2),
                Line2D([0], [0], color='black', lw=2),
               Line2D([0], [0], color='r', lw=2)]

fig.legend(custom_lines, ["DQN", "PT-DQN-0.5x (ours)", "DQN-multi-head", "DQN-large buffer", "Random"], ncol=5, fontsize=14,\
           loc="lower center", bbox_to_anchor=(0.35, 0.15, 0.4, 0.0), frameon=True)
ax.ticklabel_format(axis='x', style='sci', scilimits=(5,1))
ax.xaxis.major.formatter._useMathText = True
ax.set_xlabel("Steps", fontsize=14)
ax.set_ylabel("Reward per step\n(averaged across 10000 steps)", fontsize=14)
ax.set_title("Jellybean world", fontsize=14)
ax.tick_params(labelsize=14)
fig.tight_layout()
pdf.savefig(fig, bbox_inches = 'tight', dpi=300)
plt.show()
pdf.close()