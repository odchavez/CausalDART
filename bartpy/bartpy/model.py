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
        print("enter bartpy/bartpy/model.py Model __init__")
        
        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._prediction = None
        self._initializer = initializer

        if trees is None:
            print("in if trees is None")
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            if self._initializer is not None:
                print("in self._initializer is not None")
                self._initializer.initialize_trees(self.refreshed_trees())
        else:
            print("in else trees is not None")
            self.n_trees = len(trees)
            self._trees = trees
        print("-exit bartpy/bartpy/model.py Model __init__")

    def initialize_trees(self) -> List[Tree]:
        print("enter bartpy/bartpy/model.py Model initialize_trees")
        
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees))
        print("-exit bartpy/bartpy/model.py Model initialize_trees")
        return trees

    def residuals(self) -> np.ndarray:
        print("enter bartpy/bartpy/model.py Model residuals")
        output = self.data.y.values - self.predict()
        print("-exit bartpy/bartpy/model.py Model residuals")
        return output

    def unnormalized_residuals(self) -> np.ndarray:
        print("enter bartpy/bartpy/model.py Model unnormalized_residuals")
        output = self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())
        print("-exit bartpy/bartpy/model.py Model unnormalized_residuals")
        return output

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        print("enter bartpy/bartpy/model.py Model")
        
        if X is not None:
            output = self._out_of_sample_predict(X)
            print("-exit bartpy/bartpy/model.py Model")
            return output
        output = np.sum([tree.predict() for tree in self.trees], axis=0)
        print("-exit bartpy/bartpy/model.py Model")
        return output

    def _out_of_sample_predict(self, X: np.ndarray) -> np.ndarray:
        print("enter bartpy/bartpy/model.py Model")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        output = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        print("-exit bartpy/bartpy/model.py Model")
        return output

    @property
    def trees(self) -> List[Tree]:
        print("enter bartpy/bartpy/model.py Model")
        print("-exit bartpy/bartpy/model.py Model")
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:
        print("enter bartpy/bartpy/model.py Model")
        
        if self._prediction is None:
            self._prediction = self.predict()
        for tree in self._trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y.values - self._prediction)
            yield tree
            self._prediction += tree.predict()
        print("-exit bartpy/bartpy/model.py Model")

    @property
    def sigma_m(self) -> float:
        print("enter bartpy/bartpy/model.py Model")
        output = 0.5 / (self.k * np.power(self.n_trees, 0.5))
        print("-exit bartpy/bartpy/model.py Model")
        return output

    @property
    def sigma(self) -> Sigma:
        print("enter bartpy/bartpy/model.py Model")
        print("-exit bartpy/bartpy/model.py Model")
        return self._sigma


def deep_copy_model(model: Model) -> Model:
    print("enter bartpy/bartpy/model.py deep_copy_model")
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    print("-exit bartpy/bartpy/model.py deep_copy_model")
    return copied_model
