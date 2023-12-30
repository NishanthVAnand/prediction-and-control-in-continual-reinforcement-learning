#!/bin/bash

python PT_agent_discretegrid.py --env="DiscreteGrid" --t-episodes=500 --switch=100 --lr1=3e-4 --lr2=1e-2 --t-seeds=30 --save
python td_agent_discretegrid.py --env="DiscreteGrid" --t-episodes=500 --switch=100 --lr=0.03 --t-seeds=30 --save
python td_agent_discretegrid.py --env="DiscreteGrid" --t-episodes=500 --switch=100 --lr=0.03 --t-seeds=30 --save --reset

python plots/plot_discretegrid.py