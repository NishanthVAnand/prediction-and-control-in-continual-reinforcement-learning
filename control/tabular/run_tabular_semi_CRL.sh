#!/bin/bash

python PT_q_learning.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=300 --switch=50 --feat-type="gridTabular" --lr1=0.01 --lr2=0.5 --save
python q_learning.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=300 --switch=50 --feat-type="gridTabular" --lr1=0.5 --save
python q_learning.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=300 --switch=50 --feat-type="gridTabular" --lr1=0.1 --save --reset

python plots/plot_tabular_semi_crl.py