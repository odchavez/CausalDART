"""
A model for running many instances of BartPy models in parallel
"""
from copy import deepcopy
from typing import List

import numpy as np
from joblib import Parallel

from bartpy.bartpy.samplers.modelsampler import Chain
from bartpy.bartpy.sklearnmodel import SklearnModel


def convert_chains_models(model: SklearnModel,
                          X_s: List[np.ndarray],
                          y_s: List[np.ndarray],
                          chains: List[Chain]) -> List[SklearnModel]:
    print("enter bartpy/bartpy/runner.py convert_chains_models")
    n_chains = model.n_chains

    grouped_chains = []
    for i, x in enumerate(chains):
        if i % n_chains == 0:
            grouped_chains.append([])
        grouped_chains[-1].append(x)
    
    output = [model.from_extract(chain, x, y) for (chain, (x, y)) in zip(grouped_chains, zip(X_s, y_s))]
    print("-exit bartpy/bartpy/runner.py convert_chains_models")
    return output


def run_models(model: SklearnModel, X_s: List[np.ndarray], y_s: List[np.ndarray]) -> List[SklearnModel]:
    """
    Run an SklearnModel against a list of different data sets in parallel
    Useful coordination method when running permutation tests or cross validation

    Parameters
    ----------
    model: SklearnModel
        Base model containing parameters of the run
    X_s: List[np.ndarray]
        List of covariate matrices to run model on
    y_s: List[np.ndarray]
        List of target arrays to run model on

    Returns
    -------
    List[SklearnModel]
        List of trained SklearnModels for each of the input data sets
    """
    print("enter bartpy/bartpy/runner.py run_models")
    delayed_chains = []
    for X, y in zip(X_s, y_s):
        permuted_model = deepcopy(model)
        delayed_chains += permuted_model.f_delayed_chains(X, y)

    n_jobs = model.n_jobs
    chains = Parallel(n_jobs)(delayed_chains)
    output = convert_chains_models(model, X_s, y_s, chains)
    print("-exit bartpy/bartpy/runner.py run_models")
    return output

