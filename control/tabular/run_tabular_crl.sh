#!/bin/bash

python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=1 --decay=0.9 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=1 --decay=0.7 --lr1=0.005 --lr2=0.8 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=1 --decay=0.5 --lr1=0.005 --lr2=0.8 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=1 --decay=0.3 --lr1=0.5 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=1 --decay=0.1 --lr1=0.5 --lr2=0.5 --save

python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=5 --decay=0.9 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=5 --decay=0.7 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=5 --decay=0.5 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=5 --decay=0.3 --lr1=0.01 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=5 --decay=0.1 --lr1=0.3 --lr2=0.5 --save

python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=10 --decay=0.9 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=10 --decay=0.7 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=10 --decay=0.5 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=10 --decay=0.3 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=10 --decay=0.1 --lr1=0.005 --lr2=0.5 --save

python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=25 --decay=0.9 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=25 --decay=0.7 --lr1=0.01 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=25 --decay=0.5 --lr1=0.01 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=25 --decay=0.3 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=25 --decay=0.1 --lr1=0.01 --lr2=0.3 --save

python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=100 --decay=0.9 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=100 --decay=0.7 --lr1=0.01 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=100 --decay=0.5 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=100 --decay=0.3 --lr1=0.005 --lr2=0.5 --save
python PT_q_learning_crl.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --k=100 --decay=0.1 --lr1=0.01 --lr2=0.5 --save

python q_learning.py --t-seeds=30 --env="DiscreteGrid-v2" --t-episodes=500 --switch=50 --feat-type="gridTabular" --lr1=0.5 --save

python plots/plot_tabular_crl.py