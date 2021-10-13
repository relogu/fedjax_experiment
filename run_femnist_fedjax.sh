#!/bin/bash
declare -a elems=(
	"3 1"
	"3 100"
	"35 1" )
for elem in "${elems[@]}"; do
    read -a strarr <<< "$elem"  # uses default whitespace IFS
	python fedjax_femnist_script.py --cuda \
			--num_rounds 2000 \
			--clients_per_round ${strarr[0]} \
			--local_epochs ${strarr[1]} \
    		--verbose
done