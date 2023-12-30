#!/bin/bash

python PT_agent_itemgrid.py --env="ItemGrid" --t-episodes=450 --switch=75 --lr1=0.001 --lr2=0.01 --t-seeds=30 --save
python td_agent_itemgrid.py --env="ItemGrid" --t-episodes=450 --switch=75 --lr=0.003 --t-seeds=30 --save
python td_agent_itemgrid.py --env="ItemGrid" --t-episodes=450 --switch=75 --lr=0.03 --t-seeds=30 --save --reset

python plots/plot_itemgrid.py