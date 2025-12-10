#! /bin/sh
#
# This script is to reproduce our results-rnd in Table 2.

#alpha_list=(0.01)
#0.02 50.0 0.05 20.0 0.1 10.0 0.2 5.0 1.0)                                                              
train_rate_list=(0.025)
val_rate_list=(0.025)



for cur_rate in "${train_rate_list[@]}"; do
    for cur_val_rate in "${val_rate_list[@]}"; do
        python3 -u training_asym_sweep.py --net PolyNet --base cheb --dataset roman_empire  --K 10 --lr1 0.01 --lr3 0.01 --wd1 1e-4 --wd3 1e-4 --dropout 0.4 --dprate 0.1 --a 0.5 --b 0.25 --alpha 2.0 --semi_rnd True  --train_rate $cur_rate --val_rate $cur_val_rate --device 0 --asym  > results-sweep/out-rnd-rate-$cur_rate-cheb-asym
        python3 -u training_asym_sweep.py --net PolyNet --base cheb --dataset amazon_ratings  --K 10 --lr1 0.01 --lr3 0.01 --wd1 1e-4 --wd3 1e-4 --dropout 0.4 --dprate 0.1 --a 0.5 --b 0.25 --alpha 2.0 --semi_rnd True  --train_rate $cur_rate --val_rate $cur_val_rate --device 0 --asym  >> results-sweep/out-rnd-rate-$cur_rate-cheb-asym
        python3 -u training_asym_sweep.py --net PolyNet --base cheb --dataset minesweeper  --K 10 --lr1 0.01 --lr3 0.01 --wd1 1e-4 --wd3 1e-4 --dropout 0.4 --dprate 0.1 --a 0.5 --b 0.25 --alpha 2.0 --semi_rnd True  --train_rate $cur_rate --val_rate $cur_val_rate --device 0 --asym  >> results-sweep/out-rnd-rate-$cur_rate-cheb-asym
        python3 -u training_asym_sweep.py --net PolyNet --base cheb --dataset questions  --K 10 --lr1 0.01 --lr3 0.01 --wd1 1e-4 --wd3 1e-4 --dropout 0.4 --dprate 0.1 --a 0.5 --b 0.25 --alpha 2.0 --semi_rnd True  --train_rate $cur_rate --val_rate $cur_val_rate --device 0 --asym  >> results-sweep/out-rnd-rate-$cur_rate-cheb-asym
        python3 -u training_asym_sweep.py --net PolyNet --base cheb --dataset tolokers  --K 10 --lr1 0.01 --lr3 0.01 --wd1 1e-4 --wd3 1e-4 --dropout 0.4 --dprate 0.1 --a 0.5 --b 0.25 --alpha 2.0 --semi_rnd True  --train_rate $cur_rate --val_rate $cur_val_rate --device 0 --asym  >> results-sweep/out-rnd-rate-$cur_rate-cheb-asym

    done
done


