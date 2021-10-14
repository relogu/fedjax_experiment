#!/bin/bash
declare -a elems=(
	"1"
	"20" )
for elem in "${elems[@]}"; do
    read -a strarr <<< "$elem"  # uses default whitespace IFS
	nohup time python3 fedjax_shakespeare_script.py --cuda \
			--n_rounds 20 \
            --clients_per_round 2 \
			--local_epochs ${strarr[0]} \
			--learning_rate 0.8 \
    		-v >> "logdir/shakespeare_${strarr[0]}_log.txt"
done
# # original LEAF experiments
# for elem in "${elems[@]}"; do
#     read -a strarr <<< "$elem"
# 	python fedjax_shakespeare_script.py --cuda \
# 			--n_rounds 80 \
#             --clients_per_round 10 \
# 			--local_epochs ${strarr[0]} \
# 			--learning_rate 0.8 \
#     		-v
# done