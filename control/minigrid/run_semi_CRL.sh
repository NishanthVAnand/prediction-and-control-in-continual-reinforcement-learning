#!/bin/bash

seeds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 )
for s in ${seeds[*]};
do
	python PT_DQN.py --env="LGrid" --t-episodes=5000 --switch=500 --lr1=3e-6 --lr2=1e-4 --seed=$s --save
	python DQN.py --env="LGrid" --t-episodes=5000 --switch=500 --lr1=3e-4 --seed=$s --save
	python DQN.py --env="LGrid" --t-episodes=5000 --switch=500 --lr1=3e-4 --seed=$s --save --reset
done

python plots/plot_semi_crl.py