#!/bin/bash  

for tree_count in {1,2,3,4,5,10,30,50,100,150,200};
    do
    for sigma_a in  {3,5,10};
        do
        for sigma_q in  {0.5,0.75,0.9,0.99};
            do
                echo "gcount:" $tree_count "---hcount:" $tree_count "sigma(a,q): (" $sigma_a "," $sigma_q ")"
                python run_experiment_CBART_MM.py --n_samples 2000 --n_burn 2000 --n_trees_h $tree_count --n_trees_g $tree_count --n_chains 4 --thin .1 --sigma_a $sigma_a --sigma_q $sigma_q --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 1 --output_path "experiment_results/B/known/CBARTMM/all_runs" --data_path "simulate_data/zaidi_data_B/" --data_file_stem "zaidi_data_B_seed=" --seed_value 0 --predictors "X0,X1,X2,X3,X4" --save_g_h_sigma 1
            done
        done
    done

#for tree_count in {1,2,3,4};
#    do
#    for sigma_a in  {3,5,10};
#        do
#        for sigma_q in  {0.5,0.75,0.9,0.99};
#            do
#                echo "gcount:" $tree_count "---hcount:" $tree_count "sigma(a,q): (" $sigma_a "," $sigma_q ")"
#                python run_experiment_CBART_MM.py --n_samples 2000 --n_burn 2000 --n_trees_h $tree_count --n_trees_g $tree_count --n_chains 4 --thin .1 --sigma_a $sigma_a --sigma_q $sigma_q --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 1 --output_path "experiment_results/B/known/CBARTMM/all_runs_var=25" --data_path "simulate_data/zaidi_data_B_var=25/" --data_file_stem "zaidi_data_B_seed=" --seed_value 0 --predictors "X0,X1,X2,X3,X4" --save_g_h_sigma 1
#            done
#        done
#    done