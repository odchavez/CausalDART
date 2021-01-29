from copy import deepcopy, copy
from typing import List, Generator, Optional

import numpy as np
import pandas as pd

from bartpy.bartpy.data import Data
from bartpy.bartpy.initializers.initializer import Initializer
from bartpy.bartpy.initializers.sklearntreeinitializer import SklearnTreeInitializer
from bartpy.bartpy.sigma import Sigma
from bartpy.bartpy.split import Split
from bartpy.bartpy.tree import Tree, LeafNode, deep_copy_tree


class Model:

    def __init__(self,
                 data: Optional[Data],
                 sigma: Sigma,
                 trees: Optional[List[Tree]]=None,
                 n_trees: int=50,
                 alpha: float=0.95,
                 beta: float=2.,
                 k: int=2.,
                 initializer: Initializer=SklearnTreeInitializer()):
        #print("enter bartpy/bartpy/model.py Model __init__")
        
        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._prediction = None
        self._initializer = initializer

        if trees is None:
            #print("in if trees is None")
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            if self._initializer is not None:
                #print("in self._initializer is not None")
                self._initializer.initialize_trees(self.refreshed_trees())
        else:
            #print("in else trees is not None")
            self.n_trees = len(trees)
            self._trees = trees
        #print("-exit bartpy/bartpy/model.py Model __init__")

    def initialize_trees(self) -> List[Tree]:
        #print("enter bartpy/bartpy/model.py Model initialize_trees")
        
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees))
        #print("-exit bartpy/bartpy/model.py Model initialize_trees")
        return trees

    def residuals(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py Model residuals")
        output = self.data.y.values - self.predict()
        #print("-exit bartpy/bartpy/model.py Model residuals")
        return output

    def unnormalized_residuals(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py Model unnormalized_residuals")
        output = self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())
        #print("-exit bartpy/bartpy/model.py Model unnormalized_residuals")
        return output

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py Model predict")
        
        if X is not None:
            output = self._out_of_sample_predict(X)
            #print("-exit bartpy/bartpy/model.py Model predict")
            return output
        output = np.sum([tree.predict() for tree in self.trees], axis=0)
        #print("-exit bartpy/bartpy/model.py Model predict")
        return output

    def _out_of_sample_predict(self, X: np.ndarray) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py Model _out_of_sample_predict")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        #print("-exit bartpy/bartpy/model.py Model _out_of_sample_predict")
        return output

    @property
    def trees(self) -> List[Tree]:
        #print("enter bartpy/bartpy/model.py Model trees")
        #print("-exit bartpy/bartpy/model.py Model trees")
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:
        #print("enter bartpy/bartpy/model.py Model refreshed_trees")
        
        if self._prediction is None:
            self._prediction = self.predict()
        for tree in self._trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y.values - self._prediction)
            yield tree
            self._prediction += tree.predict()
        #print("-exit bartpy/bartpy/model.py Model refreshed_trees")

    @property
    def sigma_m(self) -> float:
        #print("enter bartpy/bartpy/model.py Model sigma_m")
        output = 0.5 / (self.k * np.power(self.n_trees, 0.5))
        #print("-exit bartpy/bartpy/model.py Model sigma_m")
        return output

    @property
    def sigma(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py Model sigma")
        #print("-exit bartpy/bartpy/model.py Model sigma")
        return self._sigma


class ModelCGM:

    def __init__(self,
                 data: Optional[Data],
                 sigma: Sigma,
                 sigma_h: Sigma,
                 sigma_g: Sigma,
                 #trees: Optional[List[Tree]]=None,
                 trees_g: Optional[List[Tree]]=None,
                 trees_h: Optional[List[Tree]]=None,
                 n_trees: int=50,
                 n_trees_g: int=50,
                 n_trees_h: int=50,
                 alpha: float=0.95,
                 beta: float=2.,
                 k: int=2.,
                 initializer: Initializer=SklearnTreeInitializer()):
        #print("enter bartpy/bartpy/model.py ModelCGM __init__")
        #print("**********************************************")
        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._sigma_h = sigma_h
        self._sigma_g = sigma_g
        self._prediction_g = None
        self._prediction_h = None
        self._initializer = initializer
        self.n_trees = n_trees
        
        if trees_g is None:
            #print("in if trees_g is None")
            self.n_trees_g = n_trees#n_trees_g
            self._trees_g = self.initialize_trees_g()
            if self._initializer is not None:
                #print("in self._initializer is not None")
                self._initializer.initialize_trees_g(self.refreshed_trees_g())
        else:
            #print("in else trees is not None")
            self.n_trees_g = len(trees_g)
            self._trees_g = trees_g

        if trees_h is None:
            #print("in if trees_h is None")
            self.n_trees_h = n_trees#n_trees_h
            self._trees_h = self.initialize_trees_h()
            if self._initializer is not None:
                #print("in self._initializer is not None")
                self._initializer.initialize_trees_h(self.refreshed_trees_h())
        else:
            #print("in else trees is not None")
            self.n_trees_h = len(trees_h)
            self._trees_h = trees_h
        
        ##print("self._trees_g=",self._trees_g)
        ##print("self._trees_h=",self._trees_h)
        ##print("type(self._trees_g)=",type(self._trees_g))
        ##print("type(self._trees_h)=",type(self._trees_h))
        #print("-exit bartpy/bartpy/model.py ModelCGM __init__")

    def initialize_trees_g(self) -> List[Tree]:
        #print("enter bartpy/bartpy/model.py ModelCGM initialize_trees_g")
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees_g)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees_g))
        #print("-exit bartpy/bartpy/model.py ModelCGM initialize_trees_g")
        return trees
    
    def initialize_trees_h(self) -> List[Tree]:
        #print("enter bartpy/bartpy/model.py ModelCGM initialize_trees_h")
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees_h)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees_h))
        #print("-exit bartpy/bartpy/model.py ModelCGM initialize_trees_h")
        return trees

    def residuals(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM residuals")
        ##print("self.predict_g()=",self.predict_g())
        W=self.data.W.values
        p=self.data.p.values
        paw = W*p**2 + (1-W)*(1-p)**2
        pbw = W*(1-p) - p*(1-W)
        ##print("Computing Residuals with self.data.y.values=", self.data.y.values)
        ##print("mean self.data.y.values=", np.mean(self.data.y.values))
        ##print("var self.data.y.values=", np.var(self.data.y.values))
        output = self.data.y.values - self.predict_g() - pbw*self.predict_h()
        #print("-exit bartpy/bartpy/model.py ModelCGM residuals")
        return output

    def residuals_g(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM residuals_g")
        ##print("self.predict_g()=",self.predict_g())
        output = self.data.y.values - self.predict_g()
        #print("-exit bartpy/bartpy/model.py ModelCGM residuals_g")
        return output
    
    def residuals_h(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM residuals_h")
        ##print("self.predict_h()=",self.predict_h())
        output = self.data.y.values - self.predict_h()
        #print("-exit bartpy/bartpy/model.py ModelCGM residuals_h")
        return output

    def unnormalized_residuals(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM unnormalized_residuals")
        output = self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())
        #print("-exit bartpy/bartpy/model.py ModelCGM unnormalized_residuals")
        return output

    def predict(self, X: np.ndarray=None) -> np.ndarray:################################this needs to account for the mixture with
        ################################ h(x) / p or 1-p subtracted out 
        #print("enter bartpy/bartpy/model.py ModelCGM predict_g")
        #print("type(self.trees_g)=",type(self.trees_g))
        if X is not None:
            output = self._out_of_sample_predict_g(X)
            #print("-exit bartpy/bartpy/model.py ModelCGM predict_g")
            return output
        output = np.sum([tree.predict_g() for tree in self.trees_g], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM predict_g")
        return output
    
    def predict_g(self, X: np.ndarray=None) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM predict_g")
        #print("type(self.trees_g)=",type(self.trees_g))
        if X is not None:
            output = self._out_of_sample_predict_g(X)
            #print("-exit bartpy/bartpy/model.py ModelCGM predict_g")
            return output
        output = np.sum([tree.predict_g() for tree in self.trees_g], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM predict_g")
        return output
    
    def predict_h(self, X: np.ndarray=None) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM predict_h")
        #print("type(self.trees_h)=",type(self.trees_h))
        if X is not None:
            output = self._out_of_sample_predict_h(X)
            #print("-exit bartpy/bartpy/model.py ModelCGM predict_h")
            return output
        output = np.sum([tree.predict_h() for tree in self.trees_h], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM predict_h")
        return output

    def _out_of_sample_predict_g(self, X: np.ndarray) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_g")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = np.sum([tree.predict(X) for tree in self.trees_g], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_g")
        return output
    
    def _out_of_sample_predict_h(self, X: np.ndarray) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_h")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = np.sum([tree.predict(X) for tree in self.trees_h], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_h")
        return output

    @property
    def trees_g(self) -> List[Tree]:
        #print("enter bartpy/bartpy/model.py ModelCGM trees_g")
        #print("-exit bartpy/bartpy/model.py ModelCGM trees_g")
        return self._trees_g
    
    @property
    def trees_h(self) -> List[Tree]:
        #print("enter bartpy/bartpy/model.py ModelCGM trees_h")
        #print("-exit bartpy/bartpy/model.py ModelCGM trees_h")
        return self._trees_h

    def refreshed_trees_g(self) -> Generator[Tree, None, None]: # the internals of the this function will need to be thouroughly checked
        #print("enter bartpy/bartpy/model.py ModelCGM refreshed_trees_g")
        current_h_of_X = self.predict_h()
        if self._prediction_g is None:
            self._prediction_g = self.predict_g()
        for tree in self._trees_g:
            self._prediction_g -= tree.predict_g()
            W = self.data.W.values
            p = self.data.p.values
            #y_vals = self.data.y.values 
            y_vals = self.data.y.values - (W*(1-p)-(1-W)*p)*current_h_of_X
            tree.update_y(y_vals - self._prediction_g)
            yield tree
            self._prediction_g += tree.predict_g()
        #print("-exit bartpy/bartpy/model.py ModelCGM refreshed_trees_g")
        
    def refreshed_trees_h(self) -> Generator[Tree, None, None]: # the internals of the this function will need to be thouroughly checked
        #print("enter bartpy/bartpy/model.py ModelCGM refreshed_trees_h")
        current_g_of_X = self.predict_g()
        #self.previous_predict_h = self.predict_h()
        if self._prediction_h is None:
            self._prediction_h = self.predict_h()
        for tree in self._trees_h:
            self._prediction_h -= tree.predict_h()
            W = self.data.W.values
            p = self.data.p.values
            factor = (W/(1-p)) - ((1-W)/p)
            current_g_adjust = current_g_of_X
            #y_vals = self.data.y.values
            y_vals = (self.data.y.values - current_g_adjust)*factor
            tree.update_y(y_vals - self._prediction_h)
            yield tree
            self._prediction_h += tree.predict_h()
            
        #print("Unique self.predict_h() values=", np.unique(self.predict_h()))
        #print("-exit bartpy/bartpy/model.py ModelCGM refreshed_trees_h")

    #@property
    #def sigma_g_m(self) -> float:
    #    #print("enter bartpy/bartpy/model.py ModelCGM sigma_g_m")
    #    output = 0.5 / (self.k * np.power(self.n_trees_g, 0.5))
    #    #print("-exit bartpy/bartpy/model.py ModelCGM sigma_g_m")
    #    return output

    @property
    def sigma_m(self) -> float:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_m")
        output = 0.5 / (self.k * np.power(self.n_trees_h, 0.5))
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_m")
        return output
    
    @property
    def sigma_g(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_g")
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_g")
        return self._sigma_g
 
    @property
    def sigma_h(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_h")
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_h")
        return self._sigma_h
    
    @property
    def sigma(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma")
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma")
        return self._sigma


def deep_copy_model(model: Model) -> Model:
    #print("enter bartpy/bartpy/model.py deep_copy_model")
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    #print("-exit bartpy/bartpy/model.py deep_copy_model")
    return copied_model


def deep_copy_model_cgm(model: ModelCGM) -> ModelCGM:
    #print("enter bartpy/bartpy/model.py deep_copy_model_cgm")
    copied_model = ModelCGM(
        None, 
        deepcopy(model.sigma), 
        deepcopy(model.sigma_h), 
        deepcopy(model.sigma_g), 
        [deep_copy_tree(tree) for tree in model.trees_g],
        [deep_copy_tree(tree) for tree in model.trees_h],
    )
    #print("-exit bartpy/bartpy/model.py deep_copy_model_cgm")
    return copied_model