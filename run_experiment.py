"""
python run_experiment_data_B_known.py --n_samples 2000 --n_burn 2000 --n_trees 200 --n_chains 4 --thin 0.1 --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 2 --output_path "experiment_results/B/known/CBARTMM/all_runs"
"""

import numpy as np
from bartpy.bartpy.sklearnmodel import SklearnModel
from tqdm import tqdm
import simulate_data.simulate_data as sd
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Trains BART model and writes output file of posterior samples.'
    )
    parser.add_argument(
        '--n_samples', type=int,
        help='n_samples',
        required=True
    )
    parser.add_argument(
        '--n_burn', type=int,
        help='n_burn',
        required=True
    )
    parser.add_argument(
        '--n_trees', type=int,
        help='n_trees',
        required=False,
        default=0
    )
    parser.add_argument(
        '--n_trees_h', type=int,
        help='n_trees',
        required=False,
        default = 0
    )
    parser.add_argument(
        '--n_trees_g', type=int,
        help='n_trees',
        required=True,
        default = 0
    )
    parser.add_argument(
        '--n_chains', type=int,
        help='n_chains',
        required=True
    )
    parser.add_argument(
        '--thin', type=float,
        help='thin',
        required=True
    )
    parser.add_argument(
        '--alpha', type=float,
        help='alpha',
        required=True
    )
    parser.add_argument(
        '--beta', type=float,
        help='beta',
        required=True
    )
    parser.add_argument(
        '--k', type=float,
        help='k',
        required=True
    )
    parser.add_argument(
        '--N_replications', type=int,
        help='N_replications',
        required=True
    )
    parser.add_argument(
        '--n', type=int,
        help='n observations',
        required=True
    )
    #parser.add_argument(
    #    '--add_prop_score', type=int,
    #    help='n observations',
    #    required=False,
    #    default=0
    #)
    parser.add_argument(
        '--output_path', type=str,
        help='output_path',
        required=True
    )
    parser.add_argument(
        '--model_type', type=str,
        help='model_type - CBARTMM: Causal BART Mixture Model,  CJHM_f(w,x) or CJHM: Causal Jennifer Hill Model, vanilla_BART_y_i_star ',
        required=False,
        default="CBARTMM"
    )
    return parser.parse_args()


args = get_args()


output_name = (args.output_path + 
               "_n_replications=" + str(args.N_replications) + 
               "_n_samples=" + str(args.n_samples) + 
               "_n_burn=" + str(args.n_burn) + 
               "_n_trees_h=" + str(args.n_trees_h) + 
               "_n_trees_g=" + str(args.n_trees_g) + 
               "_n_chains=" + str(args.n_chains) + 
               "_thin=" + str(args.thin) + 
               "_alpha=" + str(args.alpha) + 
               "_beta=" + str(args.beta) + 
               "_k=" + str(args.k) + 
               ".npy"
)

# data
if args.output_path in [
    "experiment_results/B/known/CBARTMM/all_runs",
    "experiment_results/B/known/CJHM/all_runs",
    "experiment_results/B/known/CJHM/CJHM_f(w,x)_all_runs",
    "experiment_results/B/known/vanilla_BART_y_i_star/all_runs",
]:
    data = sd.make_zaidi_data_B(args.n)
    Y,W,X,tau,pi = sd.get_data(data,0)
    
if args.output_path in [
    "experiment_results/A/known/CBARTMM/all_runs",
    "experiment_results/A/known/CJHM/all_runs",
    "experiment_results/A/known/CJHM/CJHM_f(w,x)_all_runs",
    "experiment_results/A/known/vanilla_BART_y_i_star/all_runs",
]:
    data = sd.make_zaidi_data_A(args.n)
    Y,W,X,tau,pi = sd.get_data(data,0)
    
if args.output_path in [
    "experiment_results/B/known/CBARTMM/all_runs_with_ps",
    "experiment_results/B/known/CJHM/all_runs_with_ps",
    "experiment_results/B/known/CJHM/CJHM_f(w,x)_all_runs_with_ps",
    "experiment_results/B/known/vanilla_BART_y_i_star/all_runs_with_ps",
]:
    data = sd.make_zaidi_data_B(args.n)
    Y,W,X,tau,pi = sd.get_data(data,args.n,1)
    
if args.output_path in [
    "experiment_results/A/known/CBARTMM/all_runs_with_ps",
    "experiment_results/A/known/CJHM/all_runs_with_ps",
    "experiment_results/A/known/CJHM/CJHM_f(w,x)_all_runs_with_ps",
    "experiment_results/A/known/vanilla_BART_y_i_star/all_runs_with_ps",
]:
    data = sd.make_zaidi_data_A(args.n)
d    
Y_i_star = sd.get_Y_i_star(Y,W,pi)

if args.model_type == "CBARTMM":
    # define model
    kwargs = {
        "model": "causal_gaussian_mixture"
    }
    # create the multiple instantiations of model objects
    model = []
    for i in range(args.N_replications):
        model.append(
            SklearnModel(
                n_samples=args.n_samples, 
                n_burn=args.n_burn,
                #n_trees=0,
                n_trees_h=args.n_trees_h,
                n_trees_g=args.n_trees_g,
                alpha = args.alpha, # priors for tree depth
                beta = args.beta, # priors for tree depth
                k=args.k,
                thin=args.thin,
                n_chains=args.n_chains,
                n_jobs=-1,
                store_in_sample_predictions=True,
                **kwargs
            )
        )
        
    posterior_samples = np.zeros((int(args.n_samples*args.n_chains*args.thin),args.n,args.N_replications))
    for i in tqdm(range(args.N_replications)):
        model[i].fit_CGM(X, Y_i_star, W, pi)
        posterior_samples[:,:,i]=model[i].get_posterior_CATE()

if args.model_type == "CJHM":
    # define model
    # create the multiple instantiations of model objects
    model_0 = []
    model_1 = []
    for i in range(args.N_replications):
        model_0.append(
            SklearnModel(
                n_samples=args.n_samples, 
                n_burn=args.n_burn, 
                n_trees=args.n_trees,
                alpha = args.alpha, # priors for tree depth
                beta = args.beta, # priors for tree depth
                k=args.k,
                thin=args.thin,
                n_chains=args.n_chains,
                n_jobs=-1,
                store_in_sample_predictions=True,
            )
        )
        model_1.append(
            SklearnModel(
                n_samples=args.n_samples, 
                n_burn=args.n_burn, 
                n_trees=args.n_trees,
                alpha = args.alpha, # priors for tree depth
                beta = args.beta, # priors for tree depth
                k=args.k,
                thin=args.thin,
                n_chains=args.n_chains,
                n_jobs=-1,
                store_in_sample_predictions=True,
            )
        )
        
    posterior_samples = np.zeros((args.N_replications, args.n))
    
    for i in tqdm(range(args.N_replications)):
        
        model_0[i].fit(X[W==0,:], Y[W==0])
        model_1[i].fit(X[W==1,:], Y[W==1])

        posterior_samples[i,:]= model_1[i].predict(X) - model_0[i].predict(X)


if args.model_type == "CJHM_f(w,x)":
    # define model
    # create the multiple instantiations of model objects
    model = []
    for i in range(args.N_replications):
        model.append(
            SklearnModel(
                n_samples=args.n_samples, 
                n_burn=args.n_burn, 
                n_trees=args.n_trees,
                alpha = args.alpha, # priors for tree depth
                beta = args.beta, # priors for tree depth
                k=args.k,
                thin=args.thin,
                n_chains=args.n_chains,
                n_jobs=-1,
                store_in_sample_predictions=True,
            )
        )
        
    posterior_samples = np.zeros((args.N_replications, args.n))
    for i in tqdm(range(args.N_replications)):
        X=np.concatenate([W.reshape(args.n,1), X], axis=1)
        model[i].fit(X, Y)
        X0 = X.copy()
        X[:,0]=0
        X1 = X.copy()
        X[:,0]=1
        posterior_samples[i,:]= model[i].predict(X1) - model[i].predict(X0)

        
if args.model_type == "vanilla_BART_y_i_star":
    # define model
    # create the multiple instantiations of model objects
    model = []
    for i in range(args.N_replications):
        model.append(
            SklearnModel(
                n_samples=args.n_samples, 
                n_burn=args.n_burn, 
                n_trees=args.n_trees,
                alpha = args.alpha, # priors for tree depth
                beta = args.beta, # priors for tree depth
                k=args.k,
                thin=args.thin,
                n_chains=args.n_chains,
                n_jobs=-1,
                store_in_sample_predictions=True,
            )
        )
        
    posterior_samples = np.zeros((int(args.n_samples*args.n_chains*args.thin),args.n,args.N_replications))
    for i in tqdm(range(args.N_replications)):
        model[i].fit(X, Y_i_star)
        posterior_samples[:,:,i]=model[i].get_posterior() 
        
        
print("models fit successfully")    
np.save(output_name, posterior_samples)