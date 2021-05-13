"""
python run_experiment_CBART_MM.py --n_samples 2000 --n_burn 2000 --n_trees_h 200 --n_trees_g 200 --n_chains 4 --thin 1 --alpha 0.95 --beta 2. --k 2.0 --n 250 --N_replications 1 --output_path "experiment_results/B/known/CBARTMM/all_runs" --data_path simulate_data/zaidi_data_B/ --data_file_stem zaidi_data_B_seed= --seed_value 0 --predictors X0,X1,X2,X3,X4
"""

import numpy as np
import pandas as pd
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
        '--n_trees_h', type=int,
        help='n_trees_h',
        required=False,
        default = 200
    )
    parser.add_argument(
        '--n_trees_g', type=int,
        help='n_trees_g',
        required=True,
        default = 200
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
        '--save_g_h_sigma', type=int,
        help='save all samples after burn in from model fitting including g,h and sigma',
        required=False,
        default=0
    )
    parser.add_argument(
        '--output_path', type=str,
        help='output_path',
        required=True
    )
    parser.add_argument(
        '--data_path', type=str,
        help='path of data to be loaded',
        required=True
    )
    parser.add_argument(
        '--data_file_stem', type=str,
        help='name of dat file to be loaded',
        required=True
    )
    parser.add_argument(
        '--seed_value', type=int,
        help='unique identifiier of simulation data version to be loaded.',
        required=True
    )
    parser.add_argument(
        '--predictors', type=str,
        help='List of comma separated column names WITHOUT white space characters used for predictors for X.',
        required=True
    )
    parser.add_argument(
        '--true_propensity', type=int,
        help='if 1 use True propensity score.  if 0 use estimated propensity score p_hat',
        required=False,
        default=1
    )
    parser.add_argument(
        '--fix_h', type=int,
        help='if 1 use true fixed value of h.  if 0 estimate h from data',
        required=False,
        default=0
    )
    parser.add_argument(
        '--fix_g', type=int,
        help='if 1 use true fixed value of g.  if 0 estimate g from data',
        required=False,
        default=0
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
               "_seed=" + str(args.seed_value) + 
               ".npy"
)
output_name_g = (args.output_path + 
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
               "_seed=" + str(args.seed_value) + 
               "_g.npy"
)
output_name_h = (args.output_path + 
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
               "_seed=" + str(args.seed_value) + 
               "_h.npy"
)
output_name_sigma = (args.output_path + 
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
               "_seed=" + str(args.seed_value) + 
               "_sigma.npy"
)
output_name_unnorm_sigma = (args.output_path + 
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
               "_seed=" + str(args.seed_value) + 
               "_unnorm_sigma.npy"
)
# data

data_path_and_name = args.data_path + args.data_file_stem + str(args.seed_value) + ".csv"
data = pd.read_csv(data_path_and_name)
preds = args.predictors.split(',')
X=np.array(data[preds])
Y=np.array(data["Y"])
W=np.array(data["W"])
if args.true_propensity == 1:
    p=np.array(data["p"])
else:
    p=np.array(data["p_hat"])

if args.fix_h == 1:
    fix_h = data["h(x)"]
else:
    fix_h=None
    
if args.fix_g == 1:
    fix_g = data["tau"]
else:
    fix_g=None
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
            n_trees_h=args.n_trees_h,
            n_trees_g=args.n_trees_g,
            alpha = args.alpha, # priors for tree depth
            beta = args.beta, # priors for tree depth
            k=args.k,
            thin=args.thin,
            n_chains=args.n_chains,
            n_jobs=-1,
            store_in_sample_predictions=True,
            nomalize_response_bool = False,
            **kwargs,
            fix_g=fix_g,
            fix_h=fix_h,
        )
    )
    
thinned_sample_count = int(args.n_samples*args.thin)
if args.save_g_h_sigma == 0:
    posterior_samples = np.zeros((int(thinned_sample_count*args.n_chains*args.thin),args.n,args.N_replications))
    for i in tqdm(range(args.N_replications)):
        model[i].fit_CGM(X, Y, W, p)
        posterior_samples[:,:,i]=model[i].get_posterior_CATE()
else:
    sigma = np.zeros((thinned_sample_count, args.n_chains,args.N_replications))
    unnormalized_sigma = np.zeros((thinned_sample_count, args.n_chains,args.N_replications))
    pred_g = np.zeros((args.n, thinned_sample_count, args.n_chains,args.N_replications))
    pred_h = np.zeros((args.n, thinned_sample_count, args.n_chains,args.N_replications))
    
    for i in tqdm(range(args.N_replications)):
        
        model[i].fit_CGM(X, Y, W, p)
        
        sigma_samples = np.array_split(
            [x.sigma.current_value() for x in model[i].model_samples_cgm], 
            args.n_chains
        )
        
        unnormalized_sigma_samples = np.array_split(
            [x.sigma.current_unnormalized_value() for x in model[i].model_samples_cgm], 
            args.n_chains
        )
        
        for nc in range(args.n_chains):
            sigma[:,nc,i] = sigma_samples[nc]
            unnormalized_sigma[:,nc,i] = unnormalized_sigma_samples[nc]
            pred_g[:,:,nc,i] = np.array(model[i].extract[nc]['in_sample_predictions_g']).T 
            pred_h[:,:,nc,i] = np.array(model[i].extract[nc]['in_sample_predictions_h']).T 
 
        
print("models fit successfully") 
if args.save_g_h_sigma == 0:
    np.save(output_name, posterior_samples)
else:
    print("saving g trees...")
    np.save(output_name_g, pred_g)

    print("saving h trees...")
    np.save(output_name_h, pred_h)
    
    print("saving sigma parameters...")
    np.save(output_name_sigma, sigma)
    
    print("saving unnormalized sigma parameters...")
    np.save(output_name_unnorm_sigma, unnormalized_sigma)

