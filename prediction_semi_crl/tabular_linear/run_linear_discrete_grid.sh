#!/bin/bash

python PT_Mem.py --t-seeds=30 --env=DiscreteGrid --t-episodes=500 --switch=50 --feat-type="discreteFeat" --lr1=0.001 --lr2=0.1 --save
python TD.py --t-seeds=30 --env=DiscreteGrid --t-episodes=500 --switch=50 --feat-type="discreteFeat" --lr=0.1 --save
python TD.py --t-seeds=30 --env=DiscreteGrid --t-episodes=500 --switch=50 --feat-type="discreteFeat" --lr=0.1 --reset --save

python plots/plot_linear_discrete_grid.py