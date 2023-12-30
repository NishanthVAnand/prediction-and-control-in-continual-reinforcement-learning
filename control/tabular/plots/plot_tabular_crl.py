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

def moving_average(a, n=3) :
    return np.convolve(a, np.ones((n,))/n, mode='same')

episodes = 500
switch = 50
t_seeds = 30
seeds = range(t_seeds)
gamma = 0.95
z_star = 1.645
smooth_coeff = 1
scale = 0.25

pdf = matplotlib.backends.backend_pdf.PdfPages("plots/CL_control_tabular_analysis_k_decay.pdf")
fig, ax = plt.subplots(figsize=(4,4))
mark = {1:'x', 5:'o', 10:'^', 15:'D', 25:'*', 100:'v'}

all_k = [1, 5, 10, 25, 100]
decay = [0.9, 0.7, 0.5, 0.3, 0.1]

best_lrs_all_k = {1: {0.9: (0.005, 0.5), 0.7: (0.005, 0.8), 0.5: (0.005, 0.8), 0.3: (0.5, 0.5), 0.1: (0.5, 0.5)},
5: {0.9: (0.005, 0.5), 0.7: (0.005, 0.5), 0.5: (0.005, 0.5), 0.3: (0.01, 0.5), 0.1: (0.3, 0.5)},
10: {0.9: (0.005, 0.5), 0.7: (0.005, 0.5), 0.5: (0.005, 0.5), 0.3: (0.005, 0.5), 0.1: (0.005, 0.5)},
25: {0.9: (0.005, 0.5), 0.7: (0.01, 0.5), 0.5: (0.01, 0.5), 0.3: (0.005, 0.5), 0.1: (0.01, 0.3)},
100: {0.9: (0.005, 0.5), 0.7: (0.01, 0.5), 0.5: (0.005, 0.5), 0.3: (0.005, 0.5), 0.1: (0.01, 0.5)}}

best_lr_q = (0.5, None)

for k in all_k:
    curr_k_mean = []
    curr_k_std = []
    for dec in decay:
        fname = "PT_ql_crl_ntasks_2_gamma_"+str(gamma)+"_feat_gridTabular_episodes_"+str(episodes)+"_switch_"+str(switch)+"_decay_"+str(dec)+\
        "_lr1_"+str(best_lrs_all_k[k][dec][0])+"_lr2_"+str(best_lrs_all_k[k][dec][1])+"_k_"+str(k)+"_mean.pkl"
        with open("results/DiscreteGrid-v2/"+fname, "rb") as f:
            seed_mean = pickle.load(f)
        curr_k_mean.append(seed_mean.mean())
    ax.plot(decay, curr_k_mean, lw=1.5, label=str(k), alpha=1.0, marker=mark[k], ls=":")

fname = "Q_learning_False_ntasks_2_gamma_"+str(gamma)+"_feat_gridTabular_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(best_lr_q[0])
with open("results/DiscreteGrid-v2/"+fname+"_mean.pkl", "rb") as f:
    rew = pickle.load(f)
ax.axhline(y=rew.mean(), color='k', ls='--', label="Q")
    
ax.set_xlabel(r"Decay parameter ($\lambda$)", fontsize=14)
ax.set_ylabel("Mean Returns", fontsize=14)
ax.set_xticks(decay)
ax.tick_params(labelsize=14)
ax.set_title(r"Analysis of $k$ and $\lambda$", fontsize=14)
fig.legend(ncol=1, fontsize=14, bbox_to_anchor=(0.3, 0.3, 0.97, 0.53), frameon=True, title=r"$k$", title_fontsize=14)
fig.tight_layout()
plt.grid()
pdf.savefig(fig, bbox_inches = 'tight')
plt.show()
pdf.close()