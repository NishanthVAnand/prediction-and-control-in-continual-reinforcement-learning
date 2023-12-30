import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

np.seterr(invalid='ignore')
plt.style.use('seaborn-v0_8-white')

def moving_average(a, n=3):
    cumsum_vec = np.cumsum(np.insert(a, 0, 0)) 
    ma_vec = (cumsum_vec[n:] - cumsum_vec[:-n]) / n
    return np.concatenate((a[1:n], ma_vec))

smooth_coeff = 10
z_star = 1.3

env = "LGrid"
episodes = 5000
plot_epi = 2500
switch = 500
seeds = list(range(30))
t_seeds = len(seeds)

best_lrs = {'DQN_False': (0.0003, None), 'PT_DQN': (3e-06, 0.0001), 'DQN_True': (0.0003, None)}
colors = {"PT_DQN":'b', "DQN_False":'g', "DQN_True":'r'}
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/Minigrid_control_DNN.pdf")

fig, ax = plt.subplots(figsize=(7,3.5))
for algo, (b_lr1, b_lr2) in best_lrs.items():
    seeds_returns = np.zeros((len(seeds), episodes))
    for s in seeds:
        
        if algo == "DQN_False":
            fname = "DQN_env_"+env+"_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_seed_"+str(s)+"_batch_64_reset_False_returns.pkl"
        elif algo == "PT_DQN":
            fname = "PT_DQN_env_"+env+"_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_lr2_"+str(b_lr2)+"_seed_"+str(s)+"_batch_64_returns.pkl"
        elif algo == "DQN_True":
            fname = "DQN_env_"+env+"_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_seed_"+str(s)+"_batch_64_reset_True_returns.pkl"
    
        with open("results/"+fname, "rb") as f:
            seeds_returns[s] = pickle.load(f)
    
    rew = np.mean(seeds_returns[:, :plot_epi], axis=0)
    std = np.std(seeds_returns[:, :plot_epi], axis=0)
    rew = moving_average(rew, n=smooth_coeff)
    std = moving_average(std, n=smooth_coeff)
    ax.plot(rew, label=algo, color=colors[algo], lw=1.0, alpha=0.75)
    ax.fill_between(range(plot_epi), rew+z_star*(std/t_seeds**0.5), rew-z_star*(std/t_seeds**0.5), alpha=0.2, color=colors[algo])

custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2)]

for v_cord in np.arange(switch-1, plot_epi, switch):
    plt.axvline(x=v_cord, color='k', ls=':', alpha=0.8, lw=0.6)
    
fig.legend(custom_lines, ['DQN', 'DQN (reset)', 'PT-DQN (ours)'], ncol=1, fontsize=14,\
           bbox_to_anchor=(0.65, 0.6, 0.55, 0.0), frameon=True)
ax.set_xlabel("Episodes", fontsize=14)
ax.set_ylabel("Returns", fontsize=14)
ax.set_title("Minigrid (DNN)", fontsize=14)
ax.tick_params(labelsize=14)
fig.tight_layout()
pdf.savefig(fig, bbox_inches = 'tight')
plt.show()
pdf.close()