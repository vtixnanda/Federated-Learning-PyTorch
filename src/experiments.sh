#!/bin/bash

# script to run sims of different connectivity issues
# List of values to loop over
values=(0.2 0.3 0.4 0.5 0.6 0.7)

# Loop through the values
for value in "${values[@]}"
do
    echo "Running sweep over round failure prob: $value"
    python federated_adhoc.py --model=cnn --central=1 --epochs=100 --verbose=0 --iid=0 --gpu=1 --p_round_fail=$value --num_nodes_rem=0.4 &
    python federated_adhoc.py --model=cnn --modularity=1 --epochs=100 --verbose=0 --iid=0 --gpu=1 --p_round_fail=$value --num_nodes_rem=0.4 &
    python federated_adhoc.py --model=cnn --modularity=0 --epochs=100 --verbose=0 --iid=0 --gpu=1 --p_round_fail=$value --num_nodes_rem=0.4 &
    echo "Running sweep over node percentage removed: $value"
    python federated_adhoc.py --model=cnn --central=1 --epochs=100 --verbose=0 --iid=0 --gpu=1 --num_nodes_rem=$value --p_round_fail=0.4 &
    python federated_adhoc.py --model=cnn --modularity=1 --epochs=100 --verbose=0 --iid=0 --gpu=1 --num_nodes_rem=$value --p_round_fail=0.4 &
    python federated_adhoc.py --model=cnn --modularity=0 --epochs=100 --verbose=0 --iid=0 --gpu=1 --num_nodes_rem=$value --p_round_fail=0.4 &
    wait
done