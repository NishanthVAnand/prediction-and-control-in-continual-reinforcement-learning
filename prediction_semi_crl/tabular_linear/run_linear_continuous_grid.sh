#!/bin/bash

python PT_Mem.py --t-seeds=30 --env=ContinuousGrid --t-episodes=2000 --switch=200 --feat-type="RBF" --lr1=0.005 --lr2=0.5 --save
python TD.py --t-seeds=30 --env=ContinuousGrid --t-episodes=2000 --switch=200 --feat-type="RBF" --lr=0.3 --save
python TD.py --t-seeds=30 --env=ContinuousGrid --t-episodes=2000 --switch=200 --feat-type="RBF" --lr=0.5 --reset --save

python plots/plot_linear_continuous_grid.py