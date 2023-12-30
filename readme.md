Repository to reproduce results from the paper: Prediction and Control in Continual Reinforcement Learning, NeurIPS 2023 (https://arxiv.org/abs/2312.11669).

**Note: Install all the dependencies listed in `requirements.txt` before running any of these codes.**

The code is divided into two directories: `control` and `prediction_semi_crl`, which contains the code for corresponding experiments from the paper.

## Prediction

Inside the `prediction_semi_crl` directory there are two subdirectories: `minigrid` and `tabular_linear`. `minigrid` contains the code to reproduce the results presented in the fourth and fifth subplots in Figure 2, and `tabular_linear` contains the code corresponsing to the first three subplots presented in Figure 2.

### Tabular and Linear

There are three shell scripts inside `tabular_linear`: `run_tabular.sh`, `run_linear_discrete_grid.sh`, and `run_linear_continuous_grid.sh`, which corresponds to the experiments from the first three subplots in Figure 2. Each script will run all the algorithms **sequentially** using the best hyperparameters. These scripts will produce plot PDFs inside the `plots` directory once complete.

### Minigrid

**Note: Run this code on GPU to speed up. Also, run each algorithm inside the shell scripts in parallel to further increase the speed.**

Install `gym-minigrid` using the instructions provided on https://minigrid.farama.org/index.html.

There are two shell scripts inside `minigrid`: `run_discrete_grid.sh` and `run_itemgrid.sh`, which corresponds to the experiments from the last two subplots in Figure 2. Each script will run algorithms **sequentially** using the best hyperparameters. These scripts will produce plot PDFs inside the `plots` directory once complete.

## Control

There are four directories inside `control`: `tabular`, `minigrid`, `jbw_crl`, and `minatar_crl`. `tabular` contains the code to generate tabular results presented in Figure 3b and Figure 4; `minigrid` contains the code to produce results presented in Figure 3d; `jbw_crl` contains the code to reproduce results in Figure 5b; the code inside `minatar_crl` will reproduce results presented in Figure 5d.

**Note: The shell scripts will run algorithms sequentially for each seed on a single machine. Parallelize the runs across seeds and algorithms, and them of GPUs for speeding up the execution.**

### Tabular

Run the script `run_tabular_semi_CRL.sh` to generate the plot from Figure 3b, and `run_tabular_crl.sh` to generate Figure 4. Plots are saved as PDFs inside the plots folder.

### Minigrid

Install `gym-minigrid` using the instructions provided on https://minigrid.farama.org/index.html.

Run the script `run_semi_CRL.sh` to generate the minigrid results presented in Figure 3d.

**On a GPU, each run (one seed, one algorithm, with one set of hyperparameters) took ~3 hours to complete.**

### JellyBeanWorld

Install JellyBeanWorld using the instructions provided in https://github.com/NishanthVAnand/jelly-bean-world and `jelly-bean-world` located inside `jbw_crl`. For installing jelly-bean-world, use the `setup.py` file located inside that folder for apple silicon, but use the original `setup.py` file to install it on other machines.

After installing `jelly-bean-world`, run the shell script `run_JBW.sh` to generate a PDF containing the plot presented in Figure 5b.

**On a GPU, each run (one seed, one algorithm, with one set of hyperparameters) took ~5-8 hours to complete.**

### Minatar

Install Minatar using the instruction provided in https://github.com/kenjyoung/MinAtar. After installing the dependencies, run the script `run_minatar.sh` to generate the results presented in Figure 5d.

**On a GPU, each run (one seed, one algorithm, with one set of hyperparameters) took ~5-8 hours to complete.**


## Cite this work as:

```
@inproceedings{anand2023prediction,
  title={Prediction and Control in Continual Reinforcement Learning},
  author={Anand, Nishanth and Precup, Doina},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
