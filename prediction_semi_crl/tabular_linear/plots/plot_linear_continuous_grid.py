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

best_lrs = {'TD-Res': (0.5, None), 'TD-Cont': (0.3, None), 'PT-Mem': (0.005, 0.5)}

z_star = 1.645
seeds = range(30)
t_seeds = len(seeds)
episodes = 2000
switch = 200
gamma = 0.99

pdf = matplotlib.backends.backend_pdf.PdfPages("plots/ContinuousGrid_linear_error_curves.pdf")
fig, ax = plt.subplots(figsize=(12,4))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
colors = {"PT-Mem":'b', "TD-Cont":'g', "TD-Res":'r'}

for algo, (b_lr1, b_lr2) in best_lrs.items():
    if algo == "PT-Mem":
        fname = "PT_Mem_ntasks_4_gamma_"+str(gamma)+"_feat_RBF_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_lr2_"+str(b_lr2)
    elif algo == "TD-Cont":
        fname = "TD"+"_reset_"+str(False)+"_ntasks_4_gamma_"+str(gamma)+"_feat_RBF_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)
    elif algo == "TD-Res":
        fname = "TD"+"_reset_"+str(True)+"_ntasks_4_gamma_"+str(gamma)+"_feat_RBF_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)

    with open("results/ContinuousGrid/"+fname+"_mean.pkl", "rb") as f:
        seed_mean = pickle.load(f)

    with open("results/ContinuousGrid/"+fname+"_std.pkl", "rb") as f:
        seed_std = pickle.load(f)

    with open("results/ContinuousGrid/"+fname+"_ot_mean.pkl", "rb") as f:
        seed_ot_mean = pickle.load(f)

    with open("results/ContinuousGrid/"+fname+"_ot_std.pkl", "rb") as f:
        seed_ot_std = pickle.load(f)

    plt.plot(seed_mean, c=colors[algo], alpha=0.75)
    plt.fill_between(range(episodes), seed_mean+z_star*(seed_std/t_seeds**0.5), seed_mean-z_star*(seed_std/t_seeds**0.5), alpha=0.1, color=colors[algo])
    plt.plot(seed_ot_mean, c=colors[algo], ls="--", lw=0.5, alpha=0.75)
    plt.fill_between(range(episodes), seed_ot_mean+z_star*(seed_ot_std/t_seeds**0.5), seed_ot_mean-z_star*(seed_ot_std/t_seeds**0.5), alpha=0.1, color=colors[algo])

custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='k', ls="--", lw=0.7)]
for v_cord in np.arange(switch-1, episodes, switch):
    plt.axvline(x=v_cord, color='k', ls=':', alpha=0.8, lw=0.6)
plt.legend(custom_lines, ["TD", "TD (w/ Reset)", "PT-TD (ours)", "Other Tasks"], ncol=4, fontsize=14, loc="upper right", bbox_to_anchor=(0.2, 0., 0.8, 1.0), frameon=True)
plt.xlabel("Episodes", fontsize=14)
plt.ylabel("RMSVE", fontsize=14)
plt.title("Continuous Grid (Linear)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=13)
pdf.savefig(fig, bbox_inches = 'tight')
plt.show()
pdf.close()