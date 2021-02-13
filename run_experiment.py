"""
python run_experiment_data_B_known.py --n_samples 2000 --n_burn 2000 --n_trees 200 --n_chains 4 --thin 0.1 --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 2 --output_path "experiment_results/B/known/all_runs"
"""

import numpy as np
from bartpy.bartpy.sklearnmodel import SklearnModel
from tqdm import tqdm
import simulate_data.simulate_data as sd
import argparse

def get_data(data, add_prop_score=0):
    X = data["X"]
    pi=data["p"]
    if add_prop_score == 1:
        X=np.concatenate([X, pi.reshape(n,1)], axis=1)
    return data["Y"], data["W"], X, data["tau"], pi

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
        required=True
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
    parser.add_argument(
        '--add_prop_score', type=int,
        help='n observations',
        required=False,
        default=0
    )
    parser.add_argument(
        '--output_path', type=str,
        help='output_path',
        required=True
    )
    return parser.parse_args()


args = get_args()


output_name = (args.output_path + 
               "_n_samples=" + str(args.n_samples) + 
               "_n_burn=" + str(args.n_burn) + 
               "_n_trees=" + str(args.n_trees) + 
               "_n_chains=" + str(args.n_chains) + 
               "_thin=" + str(args.thin) + 
               "_alpha=" + str(args.alpha) + 
               "_beta=" + str(args.beta) + 
               "_k=" + str(args.k) + 
               ".npy"
)

# data
if args.output_path == "experiment_results/B/known/all_runs":
    data = sd.make_zaidi_data_B(args.n)
    Y,W,X,tau,pi = get_data(data,0)
    
if args.output_path == "experiment_results/A/known/all_runs":
    data = sd.make_zaidi_data_A(args.n)
    Y,W,X,tau,pi = get_data(data,0)

if args.output_path == "experiment_results/B/unknown/all_runs":
    data = sd.make_zaidi_data_B(args.n)
    Y,W,X,tau,pi = get_data(data,1)
    
if args.output_path == "experiment_results/A/unknown/all_runs":
    data = sd.make_zaidi_data_A(args.n)
    Y,W,X,tau,pi = get_data(data,1)

Y_i_star = sd.get_Y_i_star(Y,W,pi)

# define model
kwargs = {
    "model": "causal_gaussian_mixture"
}
# create the multiple instantiations of model objects
model = []
model_with_p = []
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
            **kwargs
        )
    )
    
posterior_samples = np.zeros((int(args.n_samples*args.n_chains*args.thin),args.n,args.N_replications))
for i in tqdm(range(args.N_replications)):
    model[i].fit_CGM(X, Y_i_star, W, pi)
    posterior_samples[:,:,i]=model[i].get_posterior_CATE()
print("models fit successfully")    
np.save(output_name, posterior_samples)