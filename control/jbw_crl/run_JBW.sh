#!/bin/bash

seeds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 )
for s in ${seeds[*]};
do
	python PT_DQN_half.py --t-steps=2100000 --lr1=1e-7 --lr2=1e-3 --decay=0.75 --seed=$s --save
	python DQN.py --t-steps=2100000 --lr1=1e-4 --seed=$s --save
	python DQN_multi_head.py --t-steps=2100000 --lr1=1e-4 --seed=$s --save
	python DQN_large_buffer.py --t-steps=2100000 --lr1=1e-4 --seed=$s --save
	python random_act.py --t-steps=2100000 --seed=$s --save
done

python plots/plot_jbw.py