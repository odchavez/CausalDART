from operator import gt, le
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from bartpy.bartpy.errors import NoSplittableVariableException
from bartpy.bartpy.splitcondition import SplitCondition


def is_not_constant(series: np.ndarray) -> bool:
    """
    Quickly identify whether a series contains more than 1 distinct value
    Parameters
    ----------
    series: np.ndarray
    The series to assess

    Returns
    -------
    bool
        True if more than one distinct value found
    """
    #print("enter bartpy/bartpy/data.py is_not_constant")
    
    if len(series) <= 1:
        #print("-exit bartpy/bartpy/data.py is_not_constant")
        return False
    first_value = None
    for i in range(1, len(series)):
        # if not series.mask[i] and series.data[i] != first_value:
        if series[i] != first_value:
            if first_value is None:
                first_value = series.data[i]
            else:
                #print("-exit bartpy/bartpy/data.py is_not_constant")
                return True
    #print("-exit bartpy/bartpy/data.py is_not_constant")
    return False


def ensure_numpy_array(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    #print("enter bartpy/bartpy/data.py ensure_numpy_array")
    
    if isinstance(X, pd.DataFrame):
        #print("-exit bartpy/bartpy/data.py ensure_numpy_array")
        return X.values
    else:
        #print("-exit bartpy/bartpy/data.py ensure_numpy_array")
        return X


def ensure_float_array(X: np.ndarray) -> np.ndarray:
    #print("enter bartpy/bartpy/data.py ensure_float_array")
    #print("-exit bartpy/bartpy/data.py ensure_float_array")
    return X.astype(float)


def format_covariate_matrix(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    #print("enter bartpy/bartpy/data.py format_covariate_matrix")

    X = ensure_numpy_array(X)
    output = ensure_float_array(X)
    #print("-exit bartpy/bartpy/data.py format_covariate_matrix")
    return output


def make_bartpy_data(X: Union[np.ndarray, pd.DataFrame],
                     y: np.ndarray,
                     normalize: bool=True) -> 'Data':
    #print("enter bartpy/bartpy/data.py make_bartpy_data")
    
    X = format_covariate_matrix(X)
    y = y.astype(float)
    output = Data(X, y, normalize=normalize)
    #print("-exit bartpy/bartpy/data.py make_bartpy_data")
    return output


class CovariateMatrix(object):

    def __init__(self,
                 X: np.ndarray,
                 mask: np.ndarray,
                 n_obsv: int,
                 unique_columns: List[int],
                 splittable_variables: List[int]):
        #print("enter bartpy/bartpy/data.py CovariateMatrix __init__")
        
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values

        self._X = X
        self._n_obsv = n_obsv
        self._n_features = X.shape[1]
        self._mask = mask

        # Cache iniialization
        if unique_columns is not None:
            self._unique_columns = [x if x is True else None for x in unique_columns]
        else:
            self._unique_columns = [None for _ in range(self._n_features)]
        if splittable_variables is not None:
            self._splittable_variables = [x if x is False else None for x in splittable_variables]
        else:
            self._splittable_variables = [None for _ in range(self._n_features)]
        self._max_values = [None] * self._n_features
        self._X_column_cache = [None] * self._n_features
        self._max_value_cache = [None] * self._n_features
        self._X_cache = None
        #print("-exit bartpy/bartpy/data.py CovariateMatrix __init__")

    @property
    def mask(self) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py CovariateMatrix mask")
        #print("-exit bartpy/bartpy/data.py CovariateMatrix mask")
        return self._mask

    @property
    def values(self) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py CovariateMatrix values")
        #print("-exit bartpy/bartpy/data.py CovariateMatrix values")
        return self._X

    def get_column(self, i: int) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py CovariateMatrix get_column")
        
        if self._X_cache is None:
            self._X_cache = self.values[~self.mask, :]
        #print("-exit bartpy/bartpy/data.py CovariateMatrix get_column")
        return self._X_cache[:, i]

    def splittable_variables(self) -> List[int]:
        """
        List of columns that can be split on, i.e. that have more than one unique value

        Returns
        -------
        List[int]
            List of column numbers that can be split on
        """
        #print("enter bartpy/bartpy/data.py CovariateMatrix splittable_variables")
        
        for i in range(0, self._n_features):
            if self._splittable_variables[i] is None:
                self._splittable_variables[i] = is_not_constant(self.get_column(i))
        
        output = [i for (i, x) in enumerate(self._splittable_variables) if x is True]        
        #print("-exit bartpy/bartpy/data.py CovariateMatrix splittable_variables")
        return output

    @property
    def n_splittable_variables(self) -> int:
        #print("enter bartpy/bartpy/data.py CovariateMatrixn_splittable_variables")
        output = len(self.splittable_variables())
        #print("-exit bartpy/bartpy/data.py CovariateMatrixn_splittable_variables")
        return output

    def is_at_least_one_splittable_variable(self) -> bool:
        #print("enter bartpy/bartpy/data.py CovariateMatrix is_at_least_one_splittable_variable")
        
        if any(self._splittable_variables):
            #print("-exit bartpy/bartpy/data.py CovariateMatrix is_at_least_one_splittable_variable")
            return True
        else:
            output = len(self.splittable_variables()) > 0
            #print("-exit bartpy/bartpy/data.py CovariateMatrix is_at_least_one_splittable_variable")
            return output
    
    def random_splittable_variable(self) -> str:
        """
        Choose a variable at random from the set of splittable variables
        Returns
        -------
            str - a variable name that can be split on
        """
        #print("enter bartpy/bartpy/data.py CovariateMatrix random_splittable_variable")
        
        if self.is_at_least_one_splittable_variable():
            output = np.random.choice(np.array(self.splittable_variables()), 1)[0]
            #print("-exit bartpy/bartpy/data.py CovariateMatrix random_splittable_variable")
            return output
        else:
            raise NoSplittableVariableException()
        #print("-exit bartpy/bartpy/data.py CovariateMatrix random_splittable_variable")
        
    def is_column_unique(self, i: int) -> bool:
        """
        Identify whether feature contains only unique values, i.e. it has no duplicated values
        Useful to provide a faster way to calculate the probability of a value being selected in a variable

        Returns
        -------
        List[int]
        """
        #print("enter bartpy/bartpy/data.py CovariateMatrix is_column_unique")
        
        if self._unique_columns[i] is None:
            self._unique_columns[i] = len(np.unique(self.get_column(i))) == self._n_obsv
        output = self._unique_columns[i]
        #print("-exit bartpy/bartpy/data.py CovariateMatrix is_column_unique")
        return output

    def max_value_of_column(self, i: int):
        #print("enter bartpy/bartpy/data.py CovariateMatrix max_value_of_column")
        
        if self._max_value_cache[i] is None:
            self._max_value_cache[i] = self.get_column(i).max()
        output = self._max_value_cache[i]
        #print("-exit bartpy/bartpy/data.py CovariateMatrix max_value_of_column")
        return output

    def random_splittable_value(self, variable: int) -> Any:
        """
        Return a random value of a variable
        Useful for choosing a variable to split on

        Parameters
        ----------
        variable - str
            Name of the variable to split on

        Returns
        -------
        Any

        Notes
        -----
          - Won't create degenerate splits, all splits will have at least one row on both sides of the split
        """
        #print("enter bartpy/bartpy/data.py CovariateMatrix random_splittable_value")
        
        if variable not in self.splittable_variables():
            raise NoSplittableVariableException()
        max_value = self.max_value_of_column(variable)
        candidate = np.random.choice(self.get_column(variable))
        while candidate == max_value:
            candidate = np.random.choice(self.get_column(variable))
        #print("-exit bartpy/bartpy/data.py CovariateMatrix random_splittable_value")
        return candidate

    def proportion_of_value_in_variable(self, variable: int, value: float) -> float:
        #print("enter bartpy/bartpy/data.py CovariateMatrix proportion_of_value_in_variable")
        
        if self.is_column_unique(variable):
            output = 1. / self.n_obsv
            #print("-exit bartpy/bartpy/data.py CovariateMatrix proportion_of_value_in_variable")
            return output
        else:
            output = float(np.mean(self.get_column(variable) == value))
            #print("-exit bartpy/bartpy/data.py CovariateMatrix proportion_of_value_in_variable")
            return output

    def update_mask(self, other: SplitCondition) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py CovariateMatrix update_mask")
        
        if other.operator == gt:
            column_mask = self.values[:, other.splitting_variable] <= other.splitting_value
        elif other.operator == le:
            column_mask = self.values[:, other.splitting_variable] > other.splitting_value
        else:
            raise TypeError("Operator type not matched, only {} and {} supported".format(gt, le))
        output = self.mask | column_mask
        #print("-exit bartpy/bartpy/data.py CovariateMatrix update_mask")
        return output

    @property
    def variables(self) -> List[int]:
        #print("enter bartpy/bartpy/data.py CovariateMatrix variables")
        output = list(range(self._n_features))
        #print("-exit bartpy/bartpy/data.py CovariateMatrix variables")
        return output

    @property
    def n_obsv(self) -> int:
        #print("enter bartpy/bartpy/data.py CovariateMatrix n_obsv")
        #print("-exit bartpy/bartpy/data.py CovariateMatrix n_obsv")
        return self._n_obsv


class Target(object):

    def __init__(self, y, mask, n_obsv, normalize, y_sum=None):
        #print("enter bartpy/bartpy/data.py Target __init__")
        
        if normalize:
            self.original_y_min, self.original_y_max = y.min(), y.max()
            self._y = self.normalize_y(y)
        else:
            self._y = y
        #print("######################################### Target._mask=", mask)
        self._mask = mask
        self._inverse_mask_int = (~self._mask).astype(int)
        self._n_obsv = n_obsv
        self.normalize = normalize
        
        if y_sum is None:
            self.y_sum_cache_up_to_date = False
            self._summed_y = None
        else:
            self.y_sum_cache_up_to_date = True
            self._summed_y = y_sum
        #print("-exit bartpy/bartpy/data.py Target __init__")

    @staticmethod
    def normalize_y(y: np.ndarray) -> np.ndarray:
        """
        Normalize y into the range (-0.5, 0.5)
        Useful for allowing the leaf parameter prior to be 0, and to standardize the sigma prior

        Parameters
        ----------
        y - np.ndarray

        Returns
        -------
        np.ndarray

        Examples
        --------
        >>> Data.normalize_y([1, 2, 3])
        array([-0.5,  0. ,  0.5])
        """
        #print("enter bartpy/bartpy/data.py Target normalize_y")
        
        y_min, y_max = np.min(y), np.max(y)
        output = -0.5 + ((y - y_min) / (y_max - y_min))
        #print("-exit bartpy/bartpy/data.py Target normalize_y")
        return output

    def unnormalize_y(self, y: np.ndarray) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py Target unnormalize_y")
        
        if self.normalize == True:
            distance_from_min = y - (-0.5)
            total_distance = (self.original_y_max - self.original_y_min)
            output = self.original_y_min + (distance_from_min * total_distance)
        else:
            output=y
        #print("-exit bartpy/bartpy/data.py Target unnormalize_y")
        return output

    @property
    def unnormalized_y(self) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py Target unnormalized_y")
        if self.normalize == True:
            output = self.unnormalize_y(self.values)
        else:
            output = self.values
        #print("-exit bartpy/bartpy/data.py Target unnormalized_y")
        return output

    @property
    def normalizing_scale(self) -> float:
        #print("enter bartpy/bartpy/data.py Target normalizing_scale")
        if self.normalize == True: 
            output = self.original_y_max - self.original_y_min
        else:
            output = 1.0
        #print("-exit bartpy/bartpy/data.py Target normalizing_scale")
        return output

    def summed_y(self) -> float:
        #print("enter bartpy/bartpy/data.py Target summed_y")
        
        if self.y_sum_cache_up_to_date:
            #print("-exit bartpy/bartpy/data.py Target summed_y")
            return self._summed_y
        else:
            self._summed_y = np.sum(self._y * self._inverse_mask_int) ############### THIS IS HOW THE MASK IS USED!!!!!!!
            self.y_sum_cache_up_to_date = True
            #print("-exit bartpy/bartpy/data.py Target summed_y")
            return self._summed_y

    def update_y(self, y) -> None:
        #print("enter bartpy/bartpy/data.py Target update_y")
        #if y is not None:
        #    #print("############################################################# len(y)=", len(y))
        #    #print("#########################################self.y_sum_cache_up_to_date=", self.y_sum_cache_up_to_date)
        self._y = y
        self.y_sum_cache_up_to_date = False
        #print("-exit bartpy/bartpy/data.py Target update_y")

    @property
    def values(self):
        #print("enter bartpy/bartpy/data.py Target values")
        #print("-exit bartpy/bartpy/data.py Target values")
        return self._y

    
class PropensityScore(object):
    
    def __init__(self, p, mask, n_obsv, p_sum=None):
        #print("enter bartpy/bartpy/data.py PropensityScore __init__")
        
        self._p = p
        #print("######################################### PropensityScore._mask=", mask)
        self._mask = mask
        self._inverse_mask_int = (~self._mask).astype(int)
        self._n_obsv = n_obsv

        if p_sum is None:
            self.p_sum_cache_up_to_date = False
            self._summed_p = None
        else:
            self.p_sum_cache_up_to_date = True
            self._summed_p = p_sum
        #print("-exit bartpy/bartpy/data.py PropensityScore __init__")

    def summed_p(self) -> float:
        #print("enter bartpy/bartpy/data.py PropensityScore summed_p")
        return np.sum(self._p * self._inverse_mask_int)
        #if self.p_sum_cache_up_to_date:
        #    #print("-exit bartpy/bartpy/data.py PropensityScore summed_p")
        #    return self._summed_p
        #else:
        #    self._summed_p = np.sum(self._p * self._inverse_mask_int)
        #    self.p_sum_cache_up_to_date = True
        #    #print("-exit bartpy/bartpy/data.py PropensityScore summed_p")
        #    return self._summed_p

    def update_p(self, p) -> None:
        #print("enter bartpy/bartpy/data.py PropensityScore update_p")
        
        self._p = p
        self.p_sum_cache_up_to_date = False
        #print("-exit bartpy/bartpy/data.py PropensityScore update_p")

    @property
    def values(self):
        #print("enter bartpy/bartpy/data.py PropensityScore values")
        #print("-exit bartpy/bartpy/data.py PropensityScore values")
        return self._p


class TreatmentAssignment(object):
    
    def __init__(self, W, mask, n_obsv, W_sum=None):
        #print("enter bartpy/bartpy/data.py TreatmentAssignment __init__")
        
        self._W = W
        #print("######################################### TreatmentAssignment._mask=", mask)
        self._mask = mask
        self._inverse_mask_int = (~self._mask).astype(int)
        self._n_obsv = n_obsv

        if W_sum is None:
            self.W_sum_cache_up_to_date = False
            self._summed_W = None
        else:
            self.W_sum_cache_up_to_date = True
            self._summed_W = W_sum
        #print("-exit bartpy/bartpy/data.py TreatmentAssignment __init__")

    def summed_W(self) -> float:
        #print("enter bartpy/bartpy/data.py TreatmentAssignment summed_W")
        
        if self.W_sum_cache_up_to_date:
            #print("-exit bartpy/bartpy/data.py TreatmentAssignment summed_W")
            return self._summed_W
        else:
            self._summed_W = np.sum(self._W * self._inverse_mask_int)
            self.W_sum_cache_up_to_date = True
            #print("-exit bartpy/bartpy/data.py TreatmentAssignment summed_W")
            return self._summed_W

    def update_W(self, W) -> None:
        #print("enter bartpy/bartpy/data.py TreatmentAssignment update_W")
        
        self._W = W
        self.W_sum_cache_up_to_date = False
        #print("-exit bartpy/bartpy/data.py TreatmentAssignment update_W")

    @property
    def values(self):
        #print("enter bartpy/bartpy/data.py TreatmentAssignment values")
        #print("-exit bartpy/bartpy/data.py TreatmentAssignment values")
        return self._W


class Data(object):
    """
    Encapsulates the data within a split of feature space.
    Primarily used to cache computations on the data for better performance

    Parameters
    ----------
    X: np.ndarray
        The subset of the covariate matrix that falls into the split
    y: np.ndarray
        The subset of the target array that falls into the split
    normalize: bool
        Whether to map the target into -0.5, 0.5
    cache: bool
        Whether to cache common values.
        You really only want to turn this off if you're not going to the resulting object for anything (e.g. when testing)
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 normalize: bool=False,
                 unique_columns: List[int]=None,
                 splittable_variables: Optional[List[Optional[bool]]]=None,
                 y_sum: float=None,
                 n_obsv: int=None,
                 W: np.ndarray=None,
                 #W_sum: float=None,
                 p: np.ndarray=None,
                 #p_sum: float=None,
                 #h_of_X: np.ndarray=None,
                 #y_tilde_g_sum: float=None,
                 #g_of_X: np.ndarray=None,
                 #y_tilde_h_sum: float=None,
                ):
        #print("enter bartpy/bartpy/data.py Data __init__")
        
        if mask is None:
            mask = np.zeros_like(y).astype(bool)
        self._mask: np.ndarray = mask

        if n_obsv is None:
            n_obsv = (~self.mask).astype(int).sum()
        self._n_obsv = n_obsv
        #print("Initializing data with n_obs = ", n_obsv)
        self._X = CovariateMatrix(X, mask, n_obsv, unique_columns, splittable_variables)
        self._y = Target(y, mask, n_obsv, normalize, y_sum)
        
        condition_1 = W is not None
        condition_2 = p is not None
        if condition_1 and condition_2: ################ NEED TO ADD THE TargetCGMg and TargetCGMH Here
            #self._y_tilde_g = TargetCGMg(
            #    y=y, 
            #    mask=mask, 
            #    n_obsv=n_obsv, 
            #    normalize=normalize, 
            #    y_tilde_g_sum=y_tilde_g_sum, 
            #    W=W, 
            #    p=p, 
            #    h_of_X=h_of_X)
            self._W = TreatmentAssignment(W, mask, n_obsv, W_sum=0) ###### WILL WANT TO PASS self._y
            self._p = PropensityScore(p, mask, n_obsv, p_sum=0)
        else:
            self._W=None
            self._p=None
        #print("-exit bartpy/bartpy/data.py Data __init__")
    
    @property
    def W(self) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py Data p")
        #print("-exit bartpy/bartpy/data.py Data p")
        return self._W
    
    @property
    def p(self) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py Data p")
        #print("-exit bartpy/bartpy/data.py Data p")
        return self._p
    
    @property
    def y(self) -> Target:
        #print("enter bartpy/bartpy/data.py Data y")
        #print("-exit bartpy/bartpy/data.py Data y")
        return self._y
    
    #@property
    #def y_tilde_g(self) -> TargetCGMg:
    #    #print("enter bartpy/bartpy/data.py Data y_tilde_g")
    #    #print("-exit bartpy/bartpy/data.py Data y_tilde_g")
    #    return self._y_tilde_g

    #@property
    #def y_tilde_h(self) -> TargetCGMH:
    #    #print("enter bartpy/bartpy/data.py Data y_tilde_h")
    #    #print("-exit bartpy/bartpy/data.py Data y_tilde_h")
    #    return self._y_tilde_h
    
    @property
    def X(self) -> CovariateMatrix:
        #print("enter bartpy/bartpy/data.py Data X")
        #print("-exit bartpy/bartpy/data.py Data X")
        return self._X

    @property
    def mask(self) -> np.ndarray:
        #print("enter bartpy/bartpy/data.py Data mask")
        #print("-exit bartpy/bartpy/data.py Data mask")
        return self._mask

    def update_y(self, y: np.ndarray) -> None:
        #print("enter bartpy/bartpy/data.py Data update_y")
        self._y.update_y(y)
        #print("-exit bartpy/bartpy/data.py Data update_y")
        
    #def update_y_tilde_g(self, y_tilde_g: np.ndarray) -> None:
    #    #print("enter bartpy/bartpy/data.py Data update_y_tilde_g")
    #    self._y_tilde_g.update_y_tilde_g(y_tilde_g)
    #    #print("-exit bartpy/bartpy/data.py Data update_y_tilde_g")
    #    
    #def update_y_tilde_h(self, y_tilde_h: np.ndarray) -> None:
    #    #print("enter bartpy/bartpy/data.py Data update_y_tilde_h")
    #    self._y_tilde_h.update_y_tilde_h(y_tilde_h)
    #    #print("-exit bartpy/bartpy/data.py Data update_y_tilde_h")
    #    
    #def update_y_tilde_g_h_function(self, h_of_X: np.ndarray) -> None:
    #    #print("enter bartpy/bartpy/data.py Data update_y_tilde_g")
    #    #self._y_tilde_g.update_y_tilde_g_h_function(h_of_X)
    #    #print("-exit bartpy/bartpy/data.py Data update_y_tilde_g")
    #    pass
    #
    #def update_y_tilde_h_g_function(self, g_of_X: np.ndarray) -> None:
    #    #print("enter bartpy/bartpy/data.py Data update_y_tilde_h_g_function")
    #    #self._y_tilde_h.update_y_tilde_h_g_function(h_of_X)
    #    #print("-exit bartpy/bartpy/data.py Data update_y_tilde_h_g_function")
    #    pass
    
    def update_p(self, p: np.ndarray) -> None:
        #print("enter bartpy/bartpy/data.py Data update_p")
        self._p.update_p(p)
        #print("-exit bartpy/bartpy/data.py Data update_p")
        
    def update_W(self, W: np.ndarray) -> None:
        #print("enter bartpy/bartpy/data.py Data update_W")
        self._W.update_W(W)
        #print("-exit bartpy/bartpy/data.py Data update_W")

    def __add__(self, other: SplitCondition) -> 'Data':
        #print("enter bartpy/bartpy/data.py Data __add__")
        updated_mask = self.X.update_mask(other)
        hasattr(self, 'W')
        if (self.W is not None) and (self.p.values is not None):
            output = Data(self.X.values,
                self.y.values,
                updated_mask,
                normalize=False,
                unique_columns=self._X._unique_columns,
                splittable_variables=self._X._splittable_variables,
                y_sum=other.carry_y_sum,
                n_obsv=other.carry_n_obsv,
                W=self.W.values,
                #W_sum=other.carry_W_sum,
                p=self.p.values,
                #p_sum=other.carry_p_sum,
                #h_of_X=other.y_tilde_g.h_of_X,
                #y_tilde_g_sum=other.carry_y_tilde_g_sum, 
                #g_of_X: np.ndarray=None, y_tilde_h_sum: float=None,
            )
        else:
            output = Data(self.X.values,
                    self.y.values,
                    updated_mask,
                    normalize=False,
                    unique_columns=self._X._unique_columns,
                    splittable_variables=self._X._splittable_variables,
                    y_sum=other.carry_y_sum,
                    n_obsv=other.carry_n_obsv)
        
        #print("##################################################### self.X.values.shape", self.X.values.shape)
        #print("-exit bartpy/bartpy/data.py Data __add__")
        return output
