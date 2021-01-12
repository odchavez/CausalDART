from typing import Type

import numpy as np

from bartpy.sklearnmodel import SklearnModel


class OLS(SklearnModel):

    def __init__(self,
                 stat_model: Type,
                 **kwargs):
        print("enter bartpy/bartpy/extensions/ols.py OLS __init__")
        self.stat_model = stat_model
        self.stat_model_fit = None
        super().__init__(**kwargs)
        print("-exit bartpy/bartpy/extensions/ols.py OLS __init__")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLS':
        print("enter bartpy/bartpy/extensions/ols.py OLS fit")
        self.stat_model_fit = self.stat_model(y, X).fit()
        SklearnModel.fit(self, X, self.stat_model_fit.resid)
        print("-exit bartpy/bartpy/extensions/ols.py OLS fit")
        return self

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        print("enter bartpy/bartpy/extensions/ols.py OLS predict")
        if X is None:
            X = self.data.X
        sm_prediction = self.stat_model_fit.predict(X)
        bart_prediction = SklearnModel.predict(self, X)
        output = sm_prediction + bart_prediction
        print("-exit bartpy/bartpy/extensions/ols.py OLS predict")
        return output
