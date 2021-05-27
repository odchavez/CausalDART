#!/bin/bash  

#for tree_count in {5,10,30,50,100,150,200};
#    do
#    for sigma_a in  {3,5,10};
#        do
#        for sigma_q in  {0.5,0.75,0.9,0.99};
#            do
#                echo "gcount:" $tree_count "---hcount:" $tree_count "sigma(a,q): (" $sigma_a "," $sigma_q ")"
#                python run_experiment_CBART_MM.py --n_samples 2000 --n_burn 2000 --n_trees_h $tree_count --n_trees_g $tree_count --n_chains 4 --thin .1 --sigma_a $sigma_a --sigma_q $sigma_q --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 1 --output_path "experiment_results/A/known/CBARTMM/all_runs_var=25" --data_path "simulate_data/zaidi_data_A_var=25/0/" --data_file_stem "zaidi_data_A_seed=" --seed_value 0 --predictors "X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31,X32,X33,X34,X35,X36,X37,X38,X39" --save_g_h_sigma 1
#            done
#        done
#    done

for sigma_a in  {3,5,10};
    do
    for sigma_q in  {0.5,0.75,0.9,0.99};
        do
            echo "gcount:40 ---hcount:40 ---sigma(a,q): (" $sigma_a "," $sigma_q ")"
            python run_experiment_CBART_MM.py --n_samples 2000 --n_burn 2000 --n_trees_h 40 --n_trees_g 40 --n_chains 4 --thin .1 --sigma_a $sigma_a --sigma_q $sigma_q --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 1 --output_path "experiment_results/A/known/CBARTMM/all_runs_var=25" --data_path "simulate_data/zaidi_data_A_var=25/0/" --data_file_stem "zaidi_data_A_seed=" --seed_value 0 --predictors "X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31,X32,X33,X34,X35,X36,X37,X38,X39" --save_g_h_sigma 1
        done
    done
    