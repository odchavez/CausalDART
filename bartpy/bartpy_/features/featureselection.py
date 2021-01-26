from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

from bartpy.diagnostics.features import null_feature_split_proportions_distribution, \
    local_thresholds, global_thresholds, is_kept, feature_split_proportions, plot_feature_proportions_against_thresholds, plot_null_feature_importance_distributions, \
    plot_feature_split_proportions
from bartpy.sklearnmodel import SklearnModel


class SelectSplitProportionThreshold(BaseEstimator, SelectorMixin):

    def __init__(self,
                 model: SklearnModel,
                 percentile: float=0.2):
        print("enter bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold __init__")
        
        self.model = deepcopy(model)
        self.percentile = percentile
        print("-exit bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold __init__")

    def fit(self, X, y):
        print("enter bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold fit")
        self.model.fit(X, y)
        self.X, self.y = X, y
        self.feature_proportions = feature_split_proportions(self.model)
        print("-exit bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold fit")
        return self

    def _get_support_mask(self):
        print("enter bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold _get_support_mask")
        output = np.array([proportion > self.percentile for proportion in self.feature_proportions.values()])
        print("-exit bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold _get_support_mask")
        return output

    def plot(self):
        print("enter bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold plot")
        output
        
        plot_feature_split_proportions(self.model)
        plt.show()
        print("-exit bartpy/bartpy/features/featureselection.py SelectSplitProportionThreshold plot")


class SelectNullDistributionThreshold(BaseEstimator, SelectorMixin):

    def __init__(self,
                 model: SklearnModel,
                 percentile: float=0.95,
                 method="local",
                 n_permutations=10,
                 n_trees=None):
        print("enter bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold __init__")
        
        if method == "local":
            self.method = local_thresholds
        elif method == "global":
            self.method = global_thresholds
        else:
            raise NotImplementedError("Currently only local and global methods are supported, found {}".format(self.method))
        self.model = deepcopy(model)
        if n_trees is not None:
            self.model.n_trees = n_trees
        self.percentile = percentile
        self.n_permutations = n_permutations
        print("-exit bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold __init__")

    def fit(self, X, y):
        print("enter bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold fit")
        
        self.model.fit(X, y)
        self.X, self.y = X, y
        self.null_distribution = null_feature_split_proportions_distribution(self.model, X, y, self.n_permutations)
        self.thresholds = self.method(self.null_distribution, self.percentile)
        self.feature_proportions = feature_split_proportions(self.model)
        print("-exit bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold fit")
        return self

    def _get_support_mask(self):
        print("enter bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold _get_support_mask")
        output = np.array(is_kept(self.feature_proportions, self.thresholds))
        print("-exit bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold _get_support_mask")
        return output

    def plot(self):
        print("enter bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plot_feature_proportions_against_thresholds(self.feature_proportions, self.thresholds, ax1)
        plot_null_feature_importance_distributions(self.null_distribution, ax2)
        plt.show()
        print("-exit bartpy/bartpy/features/featureselection.py SelectNullDistributionThreshold")
