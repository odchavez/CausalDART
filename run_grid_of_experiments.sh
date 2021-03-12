#!/bin/bash  

# loops through the run experiment .py script with various values for number of trees for g and h functions

#for ht in {10,30,50,75};#,100,150,200}; #on mac
for ht in 10 30 50 75#,100,150,200} # on linux
    #do for gt in {30,50,75,100,150,200}; # mac
    do for gt in 30 50 75 100 150 200 linux
        do 
            echo "G:" $gt " - H:" $ht
        python run_experiment.py --n_samples 25000 --n_burn 1000000 --n_trees_h ${ht} --n_trees_g ${gt}  --n_chains 4 --thin 1 --alpha .95 --beta 2. --k 2. --n 250 --N_replications 1 --output_path "experiment_results/B/known/CBARTMM/all_runs" --save_g_h_sigma 1
        done
    done

#for ht in {10,20,30};
#    do for gt in {10,30,50,70,90,110,130,150,170,190}; 
#        do 
#            echo "G:" $gt " - H:" $ht
#        python run_experiment.py --n_samples 2000 --n_burn 2000 --n_trees_h ${ht} --n_trees_g ${gt}  --n_chains 4 --thin 0.1 --alpha .95 --beta 2. --k 2. --n 250 --N_replications 5 --output_path "experiment_results/B/known/CBARTMM/all_runs"
#        done
#    done

#for ht in {10,20,30,40,50,75,100}; 
#    do for gt in {10,20,30,40,50,75,100};
#        do 
#            echo "G:" $gt " - H:" $ht
#        python run_experiment.py --n_samples 2000 --n_burn 2000 --n_trees_h ${ht} --n_trees_g ${gt}  --n_chains 4 --thin 0.1 --alpha .95 --beta 2. --k 2. --n 250 --N_replications 1 --output_path "experiment_results/A/known/CBARTMM/all_runs_with_ps"
#        done
#    done

#for ht in {10,20};
#   do for gt in {50,70,90,110,130,150,170,190};
#        do echo "G:" $gt " - H:" $ht
#        python run_experiment.py --n_samples 2000 --n_burn 2000 --n_trees_h ${ht} --n_trees_g ${gt}  --n_chains 4 --thin 0.1 --alpha .95 --beta 2. --k 2. --n 250 --N_replications 5 --output_path "experiment_results/A/known/CBARTMM/all_runs_with_ps"
#        done
#    done
    
#for ht in {10,20};
#   do for gt in {200,210,220,230,240,250};
#        do echo "G:" $gt " - H:" $ht
#        python run_experiment.py --n_samples 2000 --n_burn 2000 --n_trees_h ${ht} --n_trees_g ${gt}  --n_chains 4 --thin 0.1 --alpha .95 --beta 2. --k 2. --n 250 --N_replications 5 --output_path "experiment_results/A/known/CBARTMM/all_runs_with_ps"
#        done
#    done
    
#for ht in 10;
#   do for gt in {260,270,280,290,300};
#        do echo "G:" $gt " - H:" $ht
#        python run_experiment.py --n_samples 2000 --n_burn 2000 --n_trees_h ${ht} --n_trees_g ${gt}  --n_chains 4 --thin 0.1 --alpha .95 --beta 2. --k 2. --n 250 --N_replications 5 --output_path "experiment_results/A/known/CBARTMM/all_runs_with_ps"
#        done
#    done