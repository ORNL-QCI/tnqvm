#!/bin/bash

for((r=4;r<=4;r+=2))
do
	for((n=5;n<=85;n+=5)) 
	do 
		for i in {1..10}; do read -a memvar <<< $(python3 -m memory_profiler --precision 8 profile_tnqvm.py -n $n -r $r | grep 'qpu.execute') && echo "$r, $n, ${memvar[3]}" | tee -a stats_profile_tnqvm_${r}_REDO_rounds.csv; done
	done
done
