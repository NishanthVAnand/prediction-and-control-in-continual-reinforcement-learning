#!/bin/bash

python PT_Mem.py --t-seeds=30 --env=DiscreteGrid --t-episodes=500 --switch=50 --feat-type="gridTabular" --lr1=0.01 --lr2=0.3 --save
python TD.py --t-seeds=30 --env=DiscreteGrid --t-episodes=500 --switch=50 --feat-type="gridTabular" --lr=0.3 --save
python TD.py --t-seeds=30 --env=DiscreteGrid --t-episodes=500 --switch=50 --feat-type="gridTabular" --lr=0.3 --reset --save

python plots/plot_tabular.py