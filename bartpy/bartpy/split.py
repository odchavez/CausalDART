from copy import deepcopy
from typing import List, Optional

import numpy as np

from bartpy.bartpy.data import Data
from bartpy.bartpy.splitcondition import CombinedCondition, SplitCondition


class Split:
    """
    The Split class represents the conditioned data at any point in the decision tree
    It contains the logic for:

     - Maintaining a record of which rows of the covariate matrix are in the split
     - Being able to easily access a `Data` object with the relevant rows
     - Applying `SplitConditions` to further break up the data
    """

    def __init__(self, data: Data, combined_condition: Optional[CombinedCondition]=None):
        print("enter bartpy/bartpy/split.py Split __init__")
        self._data = data
        if combined_condition is None:
            combined_condition = CombinedCondition(self._data.X.variables, [])
        self._combined_condition = combined_condition
        print("-exit bartpy/bartpy/split.py Split __init__")
        
    @property
    def data(self):
        print("enter bartpy/bartpy/split.py Split data")
        print("-exit bartpy/bartpy/split.py Split data")
        return self._data

    def combined_condition(self):
        print("enter bartpy/bartpy/split.py Split combined_condition")
        print("-exit bartpy/bartpy/split.py Split combined_condition")
        return self._combined_condition

    def condition(self, X: np.ndarray=None) -> np.array:
        print("enter bartpy/bartpy/split.py Split condition")
        
        if X is None:
            output = ~self._data.mask
            print("-exit bartpy/bartpy/split.py Split condition")
            return output
        else:
            output = self.out_of_sample_condition(X)
            print("-exit bartpy/bartpy/split.py Split condition")
            return output

    def out_of_sample_condition(self, X: np.ndarray) -> np.ndarray:
        print("enter bartpy/bartpy/split.py Split out_of_sample_condition")
        output = self._combined_condition.condition(X)
        print("-exit bartpy/bartpy/split.py Split out_of_sample_condition")
        return output

    def out_of_sample_conditioner(self) -> CombinedCondition:
        print("enter bartpy/bartpy/split.py Split out_of_sample_conditioner")
        print("-exit bartpy/bartpy/split.py Split out_of_sample_conditioner")
        return self._combined_condition

    def __add__(self, other: SplitCondition):
        print("enter bartpy/bartpy/split.py Split __add__")
        output = Split(self._data + other,
                     self._combined_condition + other)
        print("-exit bartpy/bartpy/split.py Split __add__")
        return output

    def most_recent_split_condition(self) -> Optional[SplitCondition]:
        print("enter bartpy/bartpy/split.py Split most_recent_split_condition")
        output = self._combined_condition.most_recent_split_condition()
        print("-exit bartpy/bartpy/split.py Split most_recent_split_condition")
        return output
