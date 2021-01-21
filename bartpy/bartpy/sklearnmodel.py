from copy import deepcopy
from typing import List, Callable, Mapping, Union, Optional

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import RegressorMixin, BaseEstimator

from bartpy.bartpy.data import Data
from bartpy.bartpy.initializers.initializer import Initializer
from bartpy.bartpy.initializers.sklearntreeinitializer import SklearnTreeInitializer
from bartpy.bartpy.model import Model, ModelCGM
from bartpy.bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.bartpy.samplers.modelsampler import ModelSampler, ModelSamplerCGM, Chain
from bartpy.bartpy.samplers.schedule import SampleSchedule, SampleScheduleCGM
from bartpy.bartpy.samplers.sigma import SigmaSampler
from bartpy.bartpy.samplers.treemutation import TreeMutationSampler
from bartpy.bartpy.samplers.unconstrainedtree.treemutation import get_tree_sampler
from bartpy.bartpy.sigma import Sigma


def run_chain(model: 'SklearnModel', X: np.ndarray, y: np.ndarray):
    """
    Run a single chain for a model
    Primarily used as a building block for constructing a parallel run of multiple chains
    """
    print("enter bartpy/bartpy/sklearnmodel.py run_chain")
    model.model = model._construct_model(X, y)
    output = model.sampler.samples(model.model,
                                 model.n_samples,
                                 model.n_burn,
                                 model.thin,
                                 model.store_in_sample_predictions,
                                 model.store_acceptance_trace)
    print("-exit bartpy/bartpy/sklearnmodel.py run_chain")
    return output


def run_chain_cgm(model: 'SklearnModel', X: np.ndarray, y: np.ndarray, W: np.ndarray, p: np.ndarray):
    """
    Run a single chain for a model
    Primarily used as a building block for constructing a parallel run of multiple chains
    """
    print("enter bartpy/bartpy/sklearnmodel.py run_chain_cgm")
    model.model = model._construct_model_cgm(X, y, W, p)
    output = model.sampler.samples(model.model,
                                 model.n_samples,
                                 model.n_burn,
                                 model.thin,
                                 model.store_in_sample_predictions,
                                 model.store_acceptance_trace)
    print("-exit bartpy/bartpy/sklearnmodel.py run_chain_cgm")
    return output


def delayed_run_chain():
    print("enter bartpy/bartpy/sklearnmodel.py delayed_run_chain")
    output = run_chain
    print("-exit bartpy/bartpy/sklearnmodel.py delayed_run_chain")
    return output


def delayed_run_chain_cgm():
    print("enter bartpy/bartpy/sklearnmodel.py delayed_run_chain_cgm")
    output = run_chain_cgm
    print("-exit bartpy/bartpy/sklearnmodel.py delayed_run_chain_cgm")
    return output


class SklearnModel(BaseEstimator, RegressorMixin):
    """
    The main access point to building BART models in BartPy

    Parameters
    ----------
    n_trees: int
        the number of trees to use, more trees will make a smoother fit, but slow training and fitting
    n_chains: int
        the number of independent chains to run
        more chains will improve the quality of the samples, but will require more computation
    sigma_a: float
        shape parameter of the prior on sigma
    sigma_b: float
        scale parameter of the prior on sigma
    n_samples: int
        how many recorded samples to take
    n_burn: int
        how many samples to run without recording to reach convergence
    thin: float
        percentage of samples to store.
        use this to save memory when running large models
    p_grow: float
        probability of choosing a grow mutation in tree mutation sampling
    p_prune: float
        probability of choosing a prune mutation in tree mutation sampling
    alpha: float
        prior parameter on tree structure
    beta: float
        prior parameter on tree structure
    store_in_sample_predictions: bool
        whether to store full prediction samples
        set to False if you don't need in sample results - saves a lot of memory
    store_acceptance_trace: bool
        whether to store acceptance rates of the gibbs samples
        unless you're very memory constrained, you wouldn't want to set this to false
        useful for diagnostics
    tree_sampler: TreeMutationSampler
        Method of sampling used on trees
        defaults to `bartpy.samplers.unconstrainedtree`
    initializer: Initializer
        Class that handles the initialization of tree structure and leaf values
    n_jobs: int
        how many cores to use when computing MCMC samples
        set to `-1` to use all cores
    """

    def __init__(self,
                 n_trees: int = 200,
                 n_chains: int = 4,
                 sigma_a: float = 0.001,
                 sigma_b: float = 0.001,
                 n_samples: int = 200,
                 n_burn: int = 200,
                 thin: float = 0.1,
                 alpha: float = 0.95,
                 beta: float = 2.,
                 store_in_sample_predictions: bool=False,
                 store_acceptance_trace: bool=False,
                 tree_sampler: TreeMutationSampler=get_tree_sampler(0.5, 0.5),
                 initializer: Optional[Initializer]=None,
                 n_jobs=-1,
                 **kwargs
                ):
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel __init__")
        
        if "model" in kwargs:
            if kwargs["model"] == 'causal_gaussian_mixture':
                print("Causal Gaussian Mixture using Transformed Outcomes...")
                self.model_type = 'causal_gaussian_mixture'
                self.n_trees = n_trees
                self.n_chains = n_chains
                self.sigma_a = sigma_a
                self.sigma_b = sigma_b
                self.n_burn = n_burn
                self.n_samples = n_samples
                self.p_grow = 0.5
                self.p_prune = 0.5
                self.alpha = alpha
                self.beta = beta
                self.thin = thin
                self.n_jobs = n_jobs
                self.store_in_sample_predictions = store_in_sample_predictions
                self.store_acceptance_trace = store_acceptance_trace
                self.columns = None
                #self.tree_sampler_g = get_tree_sampler(0.5, 0.5)
                #self.tree_sampler_h = get_tree_sampler(0.5, 0.5)
                self.tree_sampler = tree_sampler
                self.initializer = initializer
                self.schedule = SampleScheduleCGM(self.tree_sampler, LeafNodeSampler(), SigmaSampler())
                self.sampler = ModelSamplerCGM(self.schedule)
                self.sigma, self.data, self.model, self._prediction_samples, self._model_samples_cgm, self.extract = [None] * 6
            
        else:
            self.model_type = 'regression'
            self.n_trees = n_trees
            self.n_chains = n_chains
            self.sigma_a = sigma_a
            self.sigma_b = sigma_b
            self.n_burn = n_burn
            self.n_samples = n_samples
            self.p_grow = 0.5
            self.p_prune = 0.5
            self.alpha = alpha
            self.beta = beta
            self.thin = thin
            self.n_jobs = n_jobs
            self.store_in_sample_predictions = store_in_sample_predictions
            self.store_acceptance_trace = store_acceptance_trace
            self.columns = None
            self.tree_sampler = tree_sampler
            self.initializer = initializer
            self.schedule = SampleSchedule(self.tree_sampler, LeafNodeSampler(), SigmaSampler())
            self.sampler = ModelSampler(self.schedule)
            self.sigma, self.data, self.model, self._prediction_samples, self._model_samples, self.extract = [None] * 6
        
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel __init__") 
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'SklearnModel':
        """
        Learn the model based on training data

        Parameters
        ----------
        X: pd.DataFrame
            training covariates
        y: np.ndarray
            training targets

        Returns
        -------
        SklearnModel
            self with trained parameter values
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel fit")

        self.model = self._construct_model(X, y)
        self.extract = Parallel(n_jobs=self.n_jobs)(self.f_delayed_chains(X, y))
        self.combined_chains = self._combine_chains(self.extract)
        self._model_samples, self._prediction_samples = self.combined_chains["model"], self.combined_chains["in_sample_predictions"]
        self._acceptance_trace = self.combined_chains["acceptance"]
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel fit")
        return self
    
    def fit_CGM(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, W: np.ndarray, p: np.ndarray) -> 'SklearnModel':
        """
        Learn the model based on training data

        Parameters
        ----------
        X: pd.DataFrame
            training covariates
        y: np.ndarray
            training targets
        W: np.ndarray
            Indicator (0 or 1) indicating Treatment Assignment
        p: np.ndarray
            propensity scores
            
        Returns
        -------
        SklearnModel
            self with trained parameter values
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel fit_CGM")
        y_i_star = y*(W-p)/(p*(1-p))
        self.model = self._construct_model_cgm(X, y_i_star, W, p) #this will need to be self._construct_model_cgm(X,y,p)
        self.extract = Parallel(n_jobs=self.n_jobs)(self.f_delayed_chains_cgm(X, y_i_star, W, p))
        self.combined_chains = self._combine_chains(self.extract)
        self._model_samples_cgm, self._prediction_samples = self.combined_chains["model"], self.combined_chains["in_sample_predictions"]
        self._acceptance_trace = self.combined_chains["acceptance"]
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel fit_CGM")
        return self

    @staticmethod
    def _combine_chains(extract: List[Chain]) -> Chain:
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _combine_chains")
        keys = list(extract[0].keys())
        combined = {}
        for key in keys:
            combined[key] = np.concatenate([chain[key] for chain in extract], axis=0)
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _combine_chains")
        return combined

    @staticmethod
    def _convert_covariates_to_data(X: np.ndarray, y: np.ndarray) -> Data:
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _convert_covariates_to_data")
        from copy import deepcopy
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = Data(deepcopy(X), deepcopy(y), normalize=True)
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _convert_covariates_to_data")
        return output
    
    @staticmethod
    def _convert_covariates_to_data_cgm(X: np.ndarray, y: np.ndarray, W:np.ndarray, p: np.ndarray) -> Data:
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _convert_covariates_to_data_cgm")
        from copy import deepcopy
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = Data(deepcopy(X), deepcopy(y), W=deepcopy(W), p=deepcopy(p) , normalize=True)
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _convert_covariates_to_data_cgm")
        return output

    def _construct_model(self, X: np.ndarray, y: np.ndarray) -> Model:
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _construct_model")
        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("Empty covariate matrix passed")
        self.data = self._convert_covariates_to_data(X, y)
        self.sigma = Sigma(self.sigma_a, self.sigma_b, self.data.y.normalizing_scale)
        self.model = Model(self.data,
                           self.sigma,
                           n_trees=self.n_trees,
                           alpha=self.alpha,
                           beta=self.beta,
                           initializer=self.initializer)
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _construct_model")
        return self.model

    def _construct_model_cgm(self, X: np.ndarray, y: np.ndarray, W:np.ndarray, p:np.ndarray) -> ModelCGM:
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _construct_model_cgm")
        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("Empty covariate matrix passed")
        self.data = self._convert_covariates_to_data_cgm(X, y, W, p)
        #self.sigma = Sigma(self.sigma_a, self.sigma_b, self.data.y.normalizing_scale)
        self.sigma_g = Sigma(self.sigma_a, self.sigma_b, self.data.y.normalizing_scale)
        self.sigma_h = Sigma(self.sigma_a, self.sigma_b, self.data.y.normalizing_scale)
        self.model = ModelCGM(self.data,
                           self.sigma_g,
                           self.sigma_h,
                           n_trees_g=self.n_trees,
                           n_trees_h=self.n_trees,
                           alpha=self.alpha,
                           beta=self.beta,
                           initializer=self.initializer)
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _construct_model_cgm")
        return self.model
    
    def f_delayed_chains(self, X: np.ndarray, y: np.ndarray):
        """
        Access point for getting access to delayed methods for running chains
        Useful for when you want to run multiple instances of the model in parallel
        e.g. when calculating a null distribution for feature importance

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array

        Returns
        -------
        List[Callable[[], ChainExtract]]
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel f_delayed_chains")
        output = [delayed(x)(self, X, y) for x in self.f_chains()]
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel f_delayed_chains")
        return output
    
    def f_delayed_chains_cgm(self, X: np.ndarray, y: np.ndarray, W:np.ndarray, p:np.ndarray):
        """
        Access point for getting access to delayed methods for running chains
        Useful for when you want to run multiple instances of the model in parallel
        e.g. when calculating a null distribution for feature importance

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array
        W: np.ndarray
            Treatment Assignment Indicators
        p: np.ndarray
            Propensity Scores
        Returns
        -------
        List[Callable[[], ChainExtract]]
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel f_delayed_chains_cgm")
        output = [delayed(x)(self, X, y, W, p) for x in self.f_chains_cgm()] # this is will run )consturct_model, NOT _construct_model_cgm
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel f_delayed_chains_cgm")
        return output

    def f_chains(self) -> List[Callable[[], Chain]]:
        """
        List of methods to run MCMC chains
        Useful for running multiple models in parallel

        Returns
        -------
        List[Callable[[], Extract]]
            List of method to run individual chains
            Length of n_chains
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel f_chains")
        output = [delayed_run_chain() for _ in range(self.n_chains)]
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel f_chains")
        return output
    
    def f_chains_cgm(self) -> List[Callable[[], Chain]]:
        """
        List of methods to run MCMC chains
        Useful for running multiple models in parallel

        Returns
        -------
        List[Callable[[], Extract]]
            List of method to run individual chains
            Length of n_chains
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel f_chains")
        output = [delayed_run_chain_cgm() for _ in range(self.n_chains)]
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel f_chains")
        return output

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        """
        Predict the target corresponding to the provided covariate matrix
        If X is None, will predict based on training covariates

        Prediction is based on the mean of all samples

        Parameters
        ----------
        X: pd.DataFrame
            covariates to predict from

        Returns
        -------
        np.ndarray
            predictions for the X covariates
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel predict")
        if X is None and self.store_in_sample_predictions:
            output = self.data.y.unnormalize_y(np.mean(self._prediction_samples, axis=0))
            print("exit bartpy/bartpy/sklearnmodel.py SklearnModel predict")
            return output
        elif X is None and not self.store_in_sample_predictions:
            
            raise ValueError(
                "In sample predictions only possible if model.store_in_sample_predictions is `True`.  Either set the parameter to True or pass a non-None X parameter")
        else:
            output = self._out_of_sample_predict(X)
            print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel predict")
            return output
        
    def predict_CATE(self, X: np.ndarray=None) -> np.ndarray:
        """
        Predict the target corresponding to the provided covariate matrix
        If X is None, will predict based on training covariates

        Prediction is based on the mean of all samples

        Parameters
        ----------
        X: pd.DataFrame
            covariates to predict from

        Returns
        -------
        np.ndarray
            predictions for the X covariates
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel predict")
        if X is None and self.store_in_sample_predictions:
            output = self.data.y.unnormalize_y(np.mean(self._prediction_samples, axis=0))
            print("exit bartpy/bartpy/sklearnmodel.py SklearnModel predict")
            return output
        elif X is None and not self.store_in_sample_predictions:
            
            raise ValueError(
                "In sample predictions only possible if model.store_in_sample_predictions is `True`.  Either set the parameter to True or pass a non-None X parameter")
        else:
            output = self._out_of_sample_predict_cate(X)
            print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel predict")
            return output
        
    def predict_response(self, X: np.ndarray=None) -> np.ndarray:
        """
        Predict the target corresponding to the provided covariate matrix
        If X is None, will predict based on training covariates

        Prediction is based on the mean of all samples

        Parameters
        ----------
        X: pd.DataFrame
            covariates to predict from

        Returns
        -------
        np.ndarray
            predictions for the X covariates
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel predict")
        if X is None and self.store_in_sample_predictions:
            output = self.data.y.unnormalize_y(np.mean(self._prediction_samples, axis=0))
            print("exit bartpy/bartpy/sklearnmodel.py SklearnModel predict")
            return output
        elif X is None and not self.store_in_sample_predictions:
            
            raise ValueError(
                "In sample predictions only possible if model.store_in_sample_predictions is `True`.  Either set the parameter to True or pass a non-None X parameter")
        else:
            output = self._out_of_sample_predict_response(X)
            print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel predict")
            return output

    def residuals(self, X=None, y=None) -> np.ndarray:
        """
        Array of error for each observation

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array

        Returns
        -------
        np.ndarray
            Error for each observation
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel residuals")
        if y is None:
            output = self.model.data.y.unnormalized_y - self.predict(X)
            print("exit bartpy/bartpy/sklearnmodel.py SklearnModel residuals")
            return output
        else:
            output = y - self.predict(X)
            print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel residuals")
            return output

    def l2_error(self, X=None, y=None) -> np.ndarray:
        """
        Calculate the squared errors for each row in the covariate matrix

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array
        Returns
        -------
        np.ndarray
            Squared error for each observation
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel l2_error")
        output = np.square(self.residuals(X, y))
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel l2_error")
        return output

    def rmse(self, X, y) -> float:
        """
        The total RMSE error of the model
        The sum of squared errors over all observations

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array

        Returns
        -------
        float
            The total summed L2 error for the model
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel rmse")
        output = np.sqrt(np.sum(self.l2_error(X, y)))
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel rmse")
        return output

    def _out_of_sample_predict(self, X):
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _out_of_sample_predict")
        output = self.data.y.unnormalize_y(np.mean([x.predict(X) for x in self._model_samples], axis=0))
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _out_of_sample_predict")
        return output
    
    def _out_of_sample_predict_cate(self, X):
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _out_of_sample_predict_cate")
        output = self.data.y.unnormalize_y(np.mean([x.predict_g(X) for x in self._model_samples_cgm], axis=0))
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _out_of_sample_predict_cate")
        return output
    
    def _out_of_sample_predict_response(self, X):
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel _out_of_sample_predict_response")
        output = self.data.y.unnormalize_y(np.mean([x.predict_h(X) for x in self._model_samples_cgm], axis=0))
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel _out_of_sample_predict_response")
        return output

    def fit_predict(self, X, y):
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel fit_predict")
        self.fit(X, y)
        if self.store_in_sample_predictions:
            output = self.predict()
            print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel fit_predict")
            return output
        else:
            output = self.predict(X)
            print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel fit_predict")
            return output

    @property
    def model_samples(self) -> List[Model]:
        """
        Array of the model as it was after each sample.
        Useful for examining for:

         - examining the state of trees, nodes and sigma throughout the sampling
         - out of sample prediction

        Returns None if the model hasn't been fit

        Returns
        -------
        List[Model]
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel model_samples")
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel model_samples")
        return self._model_samples
    
    @property
    def model_samples_cgm(self) -> List[ModelCGM]:
        """
        Array of the model as it was after each sample.
        Useful for examining for:

         - examining the state of trees, nodes and sigma throughout the sampling
         - out of sample prediction

        Returns None if the model hasn't been fit

        Returns
        -------
        List[ModelCGM]
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel model_samples")
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel model_samples")
        return self._model_samples_cgm
    
    @property
    def acceptance_trace(self) -> List[Mapping[str, float]]:
        """
        List of Mappings from variable name to acceptance rates

        Each entry is the acceptance rate of the variable in each iteration of the model

        Returns
        -------
        List[Mapping[str, float]]
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel acceptance_trace")
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel acceptance_trace")
        return self._acceptance_trace

    @property
    def prediction_samples(self) -> np.ndarray:
        """
        Matrix of prediction samples at each point in sampling
        Useful for assessing convergence, calculating point estimates etc.

        Returns
        -------
        np.ndarray
            prediction samples with dimensionality n_samples * n_points
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel prediction_samples")
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel prediction_samples")
        return self.prediction_samples

    def from_extract(self, extract: List[Chain], X: np.ndarray, y: np.ndarray) -> 'SklearnModel':
        """
        Create a copy of the model using an extract
        Useful for doing operations on extracts created in external processes like feature selection
        Parameters
        ----------
        extract: Extract
            samples produced by delayed chain methods
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target variable

        Returns
        -------
        SklearnModel
            Copy of the current model with samples
        """
        print("enter bartpy/bartpy/sklearnmodel.py SklearnModel from_extract")
        new_model = deepcopy(self)
        combined_chain = self._combine_chains(extract)
        self._model_samples, self._prediction_samples = combined_chain["model"], combined_chain["in_sample_predictions"]
        self._acceptance_trace = combined_chain["acceptance"]
        new_model.data = self._convert_covariates_to_data(X, y)
        print("-exit bartpy/bartpy/sklearnmodel.py SklearnModel from_extract")
        return new_model
