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
                 #sigma_h: Sigma,
                 #sigma_g: Sigma,
                 mu_g=None,
                 mu_h=None,
                 #trees: Optional[List[Tree]]=None,
                 trees_g: Optional[List[Tree]]=None,
                 trees_h: Optional[List[Tree]]=None,
                 #n_trees: int=50,
                 n_trees_g: int=50,
                 n_trees_h: int=50,
                 alpha: float=0.95,
                 beta: float=2.,
                 k: int=2.,
                 normalize: bool=True,
                 initializer: Initializer=SklearnTreeInitializer(),
                 **kwargs,
                ):
        #print("enter bartpy/bartpy/model.py ModelCGM __init__")
        #print("**********************************************")
        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self.nomalize_response_bool = normalize
        self._sigma = sigma
        #self._sigma_h = sigma_h
        #self._sigma_g = sigma_g
        self._mu_g=mu_g
        self._mu_h=mu_h
        self._prediction_g = None
        self._prediction_h = None
        self._initializer = initializer
        if "fix_g" in kwargs:
            self.fix_g = kwargs["fix_g"]
        else:
            self.fix_g = None
        #self.n_trees = n_trees"
        
        if trees_g is None:
            #print("in if trees_g is None")
            self.n_trees_g = n_trees_g
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
            self.n_trees_h = n_trees_h
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
        #paw = W*p**2 + (1-W)*(1-p)**2
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
        if 'self.fix_g' in locals().keys():
            output = self.fix_g
        else:
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
        if 'self.fix_h' in locals().keys():
            output = self.fix_h
        else:
            output = np.sum([tree.predict_h() for tree in self.trees_h], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM predict_h")
        return output

    def _out_of_sample_predict_g(self, X: np.ndarray) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_g")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        if 'self.fix_g' in locals().keys():
            output = self.fix_g
        else:
            output = np.sum([tree.predict(X) for tree in self.trees_g], axis=0)
        #print("-exit bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_g")
        return output
    
    def _out_of_sample_predict_h(self, X: np.ndarray) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM _out_of_sample_predict_h")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        if 'self.fix_h' in locals().keys():
            output = self.fix_h
        else:
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
        if 'self.fix_g' in locals().keys():
            return
        
        if 'self.fix_h' in locals().keys():
            current_h_of_X = self.fix_h
        else:
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
        if 'self.fix_h' in locals().keys():
            return
        
        if 'self.fix_g' in locals().keys():
            current_g_of_X = self.fix_g
        else:
            current_g_of_X = self.predict_g()

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

    #@property
    #def sigma_m(self) -> float:
    #    #print("enter bartpy/bartpy/model.py ModelCGM sigma_m")
    #    if self.nomalize_response_bool:
    #        tree_count = np.max([self.n_trees_h, self.n_trees_g])
    #        output = 0.5 / (self.k * np.power(tree_count, 0.5))
    #    else:
    #        y_range = np.max(self.data.y.values)-np.min(self.data.y.values)
    #        tree_count = np.max([self.n_trees_h, self.n_trees_g])
    #        output = 0.5*y_range / (self.k * np.power(tree_count, 0.5))
    #    #print("-exit bartpy/bartpy/model.py ModelCGM sigma_m")
    #    return output
    
    @property
    def sigma_g(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_g")
        if self.nomalize_response_bool:
            tree_count = self.n_trees_g
            output = 0.5 / (self.k * np.power(tree_count, 0.5))
        else:
            y_range = np.max(self.data.y.values)-np.min(self.data.y.values)
            tree_count = self.n_trees_g
            output = 0.5*y_range / (self.k * np.power(tree_count, 0.5))
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_g")
        return output
 
    @property
    def sigma_h(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_h")
        if self.nomalize_response_bool:
            tree_count = self.n_trees_h
            output = 0.5 / (self.k * np.power(tree_count, 0.5))
        else:
            y=self.data.y.values
            p=self.data.p.values
            W=self.data.W.values
            y_obs = y * p * (1-p) / (W-p)
            y1_over_p= (y_obs/p)*W
            y0_over_1mp= (y_obs/(1.-p))*(1-W)
            
            max_val = np.max([np.max(y1_over_p), np.max(y0_over_1mp)])
            min_val = np.min([np.min(y1_over_p), np.min(y0_over_1mp)])
            y_range = max_val-min_val
            #print(y_range)
            tree_count = self.n_trees_h
            output = 0.5*y_range / (self.k * np.power(tree_count, 0.5))
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_h")
        return output
    
    @property
    def mu_h(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_h")
        if self.nomalize_response_bool:
            output = 0.
        elif self._mu_h is None:
            y=self.data.y.values
            p=self.data.p.values
            W=self.data.W.values
            y_obs = y * p * (1-p) / (W-p)
            y1_over_p= (y_obs/p)*W
            y0_over_1mp= (y_obs/(1.-p))*(1-W)
            y1_over_p=y1_over_p[y1_over_p!=0]
            y0_over_1mp=y0_over_1mp[y0_over_1mp!=0]
            #print("np.mean(y1_over_p)=",np.mean(y1_over_p))
            #print("np.mean(y0_over_p)=",np.mean(y0_over_1mp))
            mean_y1_p=np.mean(y1_over_p)
            mean_y0_1mp=np.mean(y0_over_1mp)
            output = (mean_y1_p + mean_y0_1mp) / self.n_trees_h
        else:
            output = self._mu_h/self.n_trees_h
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_h")
        #print("mu_h=",output)
        return output

    @property
    def mu_g(self) -> Sigma:
        #print("enter bartpy/bartpy/model.py ModelCGM sigma_h")
        if self.nomalize_response_bool:
            output = 0.
        elif self._mu_g is None:
            y_bar = np.mean(self.data.y.values)
            output = y_bar / self.n_trees_g
        else:
            output = self._mu_g/self.n_trees_g
        #print("-exit bartpy/bartpy/model.py ModelCGM sigma_h")
        #print("mu_g=",output)
        return output
    
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