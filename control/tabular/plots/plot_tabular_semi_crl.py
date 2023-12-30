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

z_star = 1.645
smooth_coeff = 1
plot_episodes = 300
episodes = 300
switch = 50
t_seeds = 50
seeds = range(t_seeds)
gamma = 0.95

colors = {"PT_Mem_q_learning":'blue', "Q_learning_False":'g', "Q_learning_True":'r'}
best_lrs = {'Q_learning_False': (0.5, None), 'PT_Mem_q_learning': (0.01, 0.5), 'Q_learning_True': (0.1, None)}

pdf = matplotlib.backends.backend_pdf.PdfPages("plots/Discretegrid_control_tabular.pdf")
fig, ax = plt.subplots(figsize=(8,3.5))

for algo, (b_lr1, b_lr2) in best_lrs.items():
    if algo == "Q_learning_False":
        fname = "Q_learning_False_ntasks_2_gamma_"+str(gamma)+"_feat_gridTabular_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)
    elif algo == "PT_Mem_q_learning":
        fname = "PT_Mem_q_learning"+"_ntasks_2_gamma_"+str(gamma)+"_feat_gridTabular_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_lr2_"+str(b_lr2)
    elif algo == "Q_learning_True":
        fname = "Q_learning_True_ntasks_2_gamma_"+str(gamma)+"_feat_gridTabular_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)
    
    with open("results/DiscreteGrid-v2/"+fname+"_mean.pkl", "rb") as f:
        rew = pickle.load(f)
    with open("results/DiscreteGrid-v2/"+fname+"_std.pkl", "rb") as f:
        std = pickle.load(f)
    rew = moving_average(rew, n=smooth_coeff)
    std = moving_average(std, n=smooth_coeff)
    rew = rew[:plot_episodes]
    std = std[:plot_episodes]
    ax.plot(rew, label=algo, color=colors[algo], lw=1.0, alpha=0.75)
    ax.fill_between(range(plot_episodes), rew+z_star*(std/t_seeds**0.5), rew-z_star*(std/t_seeds**0.5), alpha=0.2, color=colors[algo])

custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2)
               ]

for v_cord in np.arange(switch-1, plot_episodes, switch):
    plt.axvline(x=v_cord, color='k', ls=':', alpha=0.8, lw=0.6)

fig.legend(custom_lines, ['Q-learning', 'Q-learning (reset)' ,'PT-Q-learning (ours)'], ncol=4, fontsize=14,\
           bbox_to_anchor=(0.5, 0.35, 0.51, 0.08), frameon=True)
ax.set_xlabel("Episodes", fontsize=14)
ax.set_ylabel("Returns", fontsize=14)
ax.tick_params(labelsize=13)
ax.set_title("Discrete Grid (Tabular)", fontsize=14)
fig.tight_layout()
pdf.savefig(fig, bbox_inches = 'tight')
plt.show()
pdf.close()