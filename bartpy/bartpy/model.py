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
        
        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._prediction = None
        self._initializer = initializer

        if trees is None:
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            if self._initializer is not None:
                self._initializer.initialize_trees(self.refreshed_trees())
        else:
            self.n_trees = len(trees)
            self._trees = trees

    def initialize_trees(self) -> List[Tree]:        
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees))
        return trees

    def residuals(self) -> np.ndarray:
        output = self.data.y.values - self.predict()
        return output

    def unnormalized_residuals(self) -> np.ndarray:
        output = self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())
        return output

    def predict(self, X: np.ndarray=None) -> np.ndarray:        
        if X is not None:
            output = self._out_of_sample_predict(X)
            return output
        output = np.sum([tree.predict() for tree in self.trees], axis=0)
        return output

    def _out_of_sample_predict(self, X: np.ndarray) -> np.ndarray:
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return output

    @property
    def trees(self) -> List[Tree]:
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:        
        if self._prediction is None:
            self._prediction = self.predict()
        for tree in self._trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y.values - self._prediction)
            yield tree
            self._prediction += tree.predict()

    @property
    def sigma_m(self) -> float:
        output = 0.5 / (self.k * np.power(self.n_trees, 0.5))
        return output

    @property
    def sigma(self) -> Sigma:
        return self._sigma


class ModelCGM:

    def __init__(self,
                 data: Optional[Data],
                 sigma: Sigma,
                 sigma_h=None,
                 sigma_g=None,
                 mu_g=None,
                 mu_h=None,
                 fix_g=None,
                 fix_h=None,
                 fix_sigma=None,
                 trees_g: Optional[List[Tree]]=None,
                 trees_h: Optional[List[Tree]]=None,
                 n_trees_g: int=50,
                 n_trees_h: int=50,
                 alpha_g=None,#: float=0.95,
                 beta_g=None, #: float=2.,
                 alpha_h=None,
                 beta_h=None,
                 k: int=2.,
                 normalize=None,
                 initializer: Initializer=SklearnTreeInitializer(),
                 **kwargs,
                ):

        self.data = deepcopy(data)
        self.alpha_g = float(alpha_g)
        self.beta_g = float(beta_g)
        self.alpha_h = float(alpha_h)
        self.beta_h = float(beta_h)
        self.k = k
        self.nomalize_response_bool = normalize
        self._sigma = sigma
        self._sigma_h = sigma_h
        self._sigma_g = sigma_g
        self._mu_g=mu_g
        self._mu_h=mu_h
        self._prediction_g = None
        self._prediction_h = None
        self._initializer = initializer
        self.kwargs=kwargs
        self.fix_g = fix_g
        self.fix_h = fix_h
        self.fix_sigma = fix_sigma
        
        if trees_g is None:
            self.n_trees_g = n_trees_g
            self._trees_g = self.initialize_trees_g()
            if self._initializer is not None:
                self._initializer.initialize_trees_g(self.refreshed_trees_g())
        else:
            self.n_trees_g = len(trees_g)
            self._trees_g = trees_g

        if trees_h is None:
            self.n_trees_h = n_trees_h
            self._trees_h = self.initialize_trees_h()
            if self._initializer is not None:
                self._initializer.initialize_trees_h(self.refreshed_trees_h())
        else:
            self.n_trees_h = len(trees_h)
            self._trees_h = trees_h
        
        #print("self._mu_g=",self._mu_g)
        #print("self.fix_g =", fix_g )
        #print("self.fix_h =", fix_h )
        
    def initialize_trees_g(self) -> List[Tree]:
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees_g)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees_g))
        return trees
    
    def initialize_trees_h(self) -> List[Tree]:
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees_h)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees_h))
        return trees

    def residuals(self) -> np.ndarray:
        #print("enter bartpy/bartpy/model.py ModelCGM residuals")
        ##print("self.predict_g()=",self.predict_g())
        W=self.data.W.values
        p=self.data.p.values
        #paw = W*p**2 + (1-W)*(1-p)**2
        pbw = W*(1-p) - p*(1-W)
        #print("Computing Residuals with self.data.y.values=", self.data.y.values[:10])
        ##print("mean self.data.y.values=", np.mean(self.data.y.values))
        ##print("var self.data.y.values=", np.var(self.data.y.values))
        output = self.data.y.values - self.predict_g() - pbw*self.predict_h()
        #print("-exit bartpy/bartpy/model.py ModelCGM residuals")
        return output

    #def residuals_g(self) -> np.ndarray:
    #    #print("enter bartpy/bartpy/model.py ModelCGM residuals_g")
    #    ##print("self.predict_g()=",self.predict_g())
    #    output = self.data.y.values - self.predict_g()
    #    #print("-exit bartpy/bartpy/model.py ModelCGM residuals_g")
    #    return output
    
    #def residuals_h(self) -> np.ndarray:
    #    #print("enter bartpy/bartpy/model.py ModelCGM residuals_h")
    #    ##print("self.predict_h()=",self.predict_h())
    #    output = self.data.y.values - self.predict_h()
    #    #print("-exit bartpy/bartpy/model.py ModelCGM residuals_h")
    #    return output

    #def unnormalized_residuals(self) -> np.ndarray:
    #    #print("enter bartpy/bartpy/model.py ModelCGM unnormalized_residuals")
    #    print("unnormalized_residuals() called ********************************************")
    #    #output = self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())
    #    #print("-exit bartpy/bartpy/model.py ModelCGM unnormalized_residuals")
    #    #return output
    #    pass

    #def predict(self, X: np.ndarray=None) -> np.ndarray:
    #    print("predict() called ********************************************")
    #    if X is not None:
    #        output = self._out_of_sample_predict_g(X)
    #        return output
    #    output = np.sum([tree.predict_g() for tree in self.trees_g], axis=0)
    #    return output
    
    def predict_g(self, X: np.ndarray=None) -> np.ndarray:
        if X is not None:
            #print("stage 1")
            if self.fix_g is None:
                #print("using trees for model.predict_g")
                output = self._out_of_sample_predict_g(X)
            else:
                #print("using fix_g for model.predict_g")
                output = self.fix_g
            return output
        
        if self.fix_g is None:
            #print("stage 2")
            #print("using trees for model.predict_g")
            output = np.sum([tree.predict_g() for tree in self.trees_g], axis=0)
        else:
            #print("using fix_g for model.predict_g")
            output = self.fix_g

        return output
    
    def predict_h(self, X: np.ndarray=None) -> np.ndarray:
        if X is not None:
            if self.fix_h is None:
                #print("using trees for predict_h")
                output = self._out_of_sample_predict_h(X)
            else:
                #print("using fix_h predict_h")
                output=self.fix_h
            return output
        if self.fix_h is None:
            #print("using trees for predict_h")
            output = np.sum([tree.predict_h() for tree in self.trees_h], axis=0)
        else:
            #print("using fix_h for predict_h")
            output=self.fix_h
        return output

    def _out_of_sample_predict_g(self, X: np.ndarray) -> np.ndarray:
        #print("enter model._out_of_sample_predict_g")
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        if self.fix_g is not None:
            #print("using fix_g for model._out_of_sample_predict_g")
            output = self.fix_g
        else:
            #print("using trees for model._out_of_sample_predict_g")
            output = np.sum([tree.predict(X) for tree in self.trees_g], axis=0)
        #print("exit model._out_of_sample_predict_g")    
        return output
    
    def _out_of_sample_predict_h(self, X: np.ndarray) -> np.ndarray:
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        if self.fix_h is not None:
            #print("using fix_g for model._out_of_sample_predict_g")
            output = self.fix_h
        else:
            output = np.sum([tree.predict(X) for tree in self.trees_h], axis=0)
        return output

    @property
    def trees_g(self) -> List[Tree]:
        #print("using model property model.trees_g")
        return self._trees_g
    
    @property
    def trees_h(self) -> List[Tree]:
        return self._trees_h

    def refreshed_trees_g(self) -> Generator[Tree, None, None]:
        #print("enter model.refreshed_trees_g")    
        
        if self.fix_g is not None:
            #print("returning but doing nothing...")
            #print("exit model.refreshed_trees_g")
            return
        
        if self.fix_h is not None:
            current_h_of_X = self.fix_h
        else:
            current_h_of_X = self.predict_h()
        
        if self._prediction_g is None:
            self._prediction_g = self.predict_g()
        #print("*******************************************")
        #print("**                  G                    **")
        #print("*******************************************")
        #tree_counter = 0
        for tree in self._trees_g:
            #tree_counter+=1
            #print("g tree:",str(tree_counter))
            self._prediction_g -= tree.predict_g()
            W = self.data.W.values
            p = self.data.p.values
            y_vals = self.data.y.values - (W*(1-p)-(1-W)*p)*current_h_of_X
            tree.update_y(y_vals - self._prediction_g)
            yield tree
            self._prediction_g += tree.predict_g()
        #print("returning after doing work in model.refreshed_trees_g")
        #print("exit model.refreshed_trees_g")
        
    def refreshed_trees_h(self) -> Generator[Tree, None, None]:
        #print("Refreshing h(x) trees...")
        if self.fix_h is not None:
            return
        
        if self.fix_g is not None:
            current_g_of_X = self.fix_g
        else:
            current_g_of_X = self.predict_g()
        
        if self._prediction_h is None:
            self._prediction_h = self.predict_h()
        #print("*******************************************")
        #print("**                  H                    **")
        #print("*******************************************")
        #tree_counter = 0
        for tree in self._trees_h:
            #tree_counter+=1
            #print("h tree:",str(tree_counter))
            self._prediction_h -= tree.predict_h() # sum of trees minus j_th tree
            W = self.data.W.values
            p = self.data.p.values
            factor = (W/(1-p)) - ((1-W)/p)
            #print("first self.data.y.values[:10]=", self.data.y.values[:10])
            #print("(self.data.y.values - current_g_of_X)*factor=",((self.data.y.values - current_g_of_X)*factor)[:10])
            y_vals = (self.data.y.values - current_g_of_X)*factor
            tree.update_y(y_vals - self._prediction_h)
            yield tree
            self._prediction_h += tree.predict_h()
    
    @property
    def sigma_g(self) -> Sigma:
        return self._sigma_g
 
    @property
    def sigma_h(self) -> Sigma:
        return self._sigma_h
    
    @property
    def mu_h(self) -> Sigma:
        return self._mu_h

    @property
    def mu_g(self) -> Sigma:
        return self._mu_g
    
    @property
    def sigma(self) -> Sigma:
        return self._sigma


def deep_copy_model(model: Model) -> Model:
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    return copied_model


def deep_copy_model_cgm(model: ModelCGM) -> ModelCGM:
    copied_model = ModelCGM(
        data=None, 
        sigma=deepcopy(model.sigma), 
        trees_g=[deep_copy_tree(tree) for tree in model.trees_g],
        trees_h=[deep_copy_tree(tree) for tree in model.trees_h],
        mu_g=model._mu_g,
        mu_h=model._mu_h,
        fix_g=model.fix_g,
        fix_h=model.fix_h,
        fix_sigma=model.fix_sigma,
        alpha_g=model.alpha_g,
        alpha_h=model.alpha_h,
        beta_g=model.beta_g,
        beta_h=model.beta_h,
    )
    return copied_model
