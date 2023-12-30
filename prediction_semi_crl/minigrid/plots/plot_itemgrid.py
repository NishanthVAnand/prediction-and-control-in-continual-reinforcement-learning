import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from configparser import ConfigParser
from matplotlib.ticker import FormatStrFormatter

np.seterr(invalid='ignore')
plt.style.use('seaborn-v0_8-white')
env = "ItemGrid"
episodes = 450
switch = 75
t_seeds = 30
z_star = 1.645
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/"+env+"_Deep_prediction_performance.pdf")
fig = plt.figure(figsize=(14, 5))
colors = {"PT-Mem":'b', "TD-Cont":'g', "TD-Res":'r'}
best_lrs = {'TD-Res': (0.03, None), 'TD-Cont': (0.003, None), 'PT-Mem': (0.001, 0.01)}

for algo, (b_lr1, b_lr2) in best_lrs.items():
    if algo == "PT-Mem":
        filename = "PT_env_"+env+"_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_lr2_"+str(b_lr2)+"_t_seeds_"+str(t_seeds)+"_batch_64"
    elif algo == "TD-Cont":
        filename = "TD_env_"+env+"_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_t_seeds_"+str(t_seeds)+"_reset_"+str(False)+"_batch_64"
    elif algo == "TD-Res":
        filename = "TD_env_"+env+"_episodes_"+str(episodes)+"_switch_"+str(switch)+"_lr1_"+str(b_lr1)+"_t_seeds_"+str(t_seeds)+"_reset_"+str(True)+"_batch_64"

    with open("results/"+filename+"_curr_errors_mean.pkl", "rb") as f:
        online_err_mean = pickle.load(f)

    with open("results/"+filename+"_oth_errors_mean.pkl", "rb") as f:
        others_err_mean = pickle.load(f)

    with open("results/"+filename+"_curr_errors_std.pkl", "rb") as f:
        online_err_std = pickle.load(f)

    with open("results/"+filename+"_oth_errors_std.pkl", "rb") as f:
        others_err_std = pickle.load(f)
    
    plt.plot(range(episodes), online_err_mean, c=colors[algo], alpha=0.75)
    plt.fill_between(range(episodes), online_err_mean+z_star*(online_err_std/t_seeds**0.5), online_err_mean-z_star*(online_err_std/t_seeds**0.5), alpha=0.1, color=colors[algo])
    plt.plot(range(episodes), others_err_mean, c=colors[algo], ls="--", lw=0.7, alpha=0.75)

custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='k', ls="--", lw=0.7)]
for v_cord in np.arange(switch-1, episodes, switch):
    plt.axvline(x=v_cord, color='k', ls=':', alpha=0.8, lw=0.6)
plt.legend(custom_lines, ["Deep TD", "Deep TD (reset)", "PT-Deep TD", "Other Tasks"], ncol=4, fontsize=14, loc="upper right", bbox_to_anchor=(0.2, 0., 0.8, 1.0), frameon=True)
plt.xlabel("Episodes", fontsize=14)
plt.ylabel("RMSVE", fontsize=14)
plt.xticks(list(range(0, episodes+1, 75)))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.title("Gym Minigrid (DNN)", fontsize=14)
pdf.savefig(fig, bbox_inches='tight')
pdf.close()
plt.show()