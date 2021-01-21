from operator import le, gt
from typing import Callable, List, Optional, Union

import numpy as np


class SplitCondition(object):
    """
    A representation of a split in feature space.
    The two main components are:

        - splitting_variable: which variable is being split on
        - splitting_value: the value being split on
                           all values less than or equal to this go left, all values greater go right

    """

    def __init__(self, 
                 splitting_variable: int, 
                 splitting_value: float, 
                 operator: Callable[[float, float], bool], 
                 condition=None,
                 carry_y_sum=None,
                 carry_n_obsv=None,
                 carry_W_sum=None,
                 carry_p_sum=None,
                 carry_y_tilde_g_sum=None,
                 carry_y_tilde_h_sum=None,):
        print("enter bartpy/bartpy/splitcondition.py SplitCondition __init__")
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._condition = condition
        self.operator = operator

        self.carry_y_sum = carry_y_sum
        self.carry_y_tilde_g_sum = carry_y_tilde_g_sum
        self.carry_y_tilde_h_sum = carry_y_tilde_h_sum
        self.carry_W_sum = carry_W_sum
        self.carry_p_sum = carry_p_sum
        self.carry_n_obsv = carry_n_obsv
        print("-exit bartpy/bartpy/splitcondition.py SplitCondition __init__")

    def __str__(self):
        print("enter bartpy/bartpy/splitcondition.py SplitCondition __str__")
        output = str(self.splitting_variable) + ": " + str(self.splitting_value)
        print("-exit bartpy/bartpy/splitcondition.py SplitCondition __str__")     
        return output

    def __eq__(self, other: 'SplitCondition'):
        print("enter bartpy/bartpy/splitcondition.py SplitCondition __eq__")
        output = self.splitting_variable == other.splitting_variable and self.splitting_value == other.splitting_value and self.operator == other.operator
        print("-exit bartpy/bartpy/splitcondition.py SplitCondition __eq__")     
        return output


class CombinedVariableCondition(object):

    def __init__(self, splitting_variable: int, min_value: float, max_value: float):
        print("enter bartpy/bartpy/splitcondition.py CombinedVariableCondition __init__")
        
        self.splitting_variable = splitting_variable
        self.min_value, self.max_value = min_value, max_value
        print("-exit bartpy/bartpy/splitcondition.py CombinedVariableCondition __init__")     

    def add_condition(self, split_condition: SplitCondition) -> 'CombinedVariableCondition':
        print("enter bartpy/bartpy/splitcondition.py CombinedVariableCondition add_condition")
        
        if self.splitting_variable != split_condition.splitting_variable:
            print("-exit bartpy/bartpy/splitcondition.py CombinedVariableCondition add_condition")     
            return self
        if split_condition.operator == gt and split_condition.splitting_value > self.min_value:
            output = CombinedVariableCondition(self.splitting_variable, split_condition.splitting_value, self.max_value)
            print("-exit bartpy/bartpy/splitcondition.py CombinedVariableCondition add_condition")     
            return output
        elif split_condition.operator == le and split_condition.splitting_value < self.max_value:
            output = CombinedVariableCondition(self.splitting_variable, self.min_value, split_condition.splitting_value)
            print("-exit bartpy/bartpy/splitcondition.py CombinedVariableCondition add_condition")     
            return output
        else:
            print("-exit bartpy/bartpy/splitcondition.py CombinedVariableCondition add_condition")     
            return self


class CombinedCondition(object):

    def __init__(self, variables: List[int], conditions: List[SplitCondition]):
        print("enter bartpy/bartpy/splitcondition.py CombinedCondition __init__")
             
        self.variables = {v: CombinedVariableCondition(v, -np.inf, np.inf) for v in variables}
        self.conditions = conditions
        for condition in conditions:
            self.variables[condition.splitting_variable] = self.variables[condition.splitting_variable].add_condition(condition)
        if len(conditions) > 0:
            self.splitting_variable = conditions[-1].splitting_variable
        else:
            self.splitting_variable = None
        print("-exit bartpy/bartpy/splitcondition.py CombinedCondition __init__")

    def condition(self, X: np.ndarray) -> np.ndarray:
        print("enter bartpy/bartpy/splitcondition.py CombinedCondition condition")
        
        c = np.array([True] * len(X))
        for variable in self.variables.keys():
            c = c & (X[:, variable] > self.variables[variable].min_value) & (X[:, variable] <= self.variables[variable].max_value)
        print("-exit bartpy/bartpy/splitcondition.py CombinedCondition condition")     
        return c

    def __add__(self, other: SplitCondition):
        print("enter bartpy/bartpy/splitcondition.py CombinedCondition __add__")
        output = CombinedCondition(list(self.variables.keys()), self.conditions + [other])
        print("-exit bartpy/bartpy/splitcondition.py CombinedCondition __add__")     
        return output

    def most_recent_split_condition(self):
        print("enter bartpy/bartpy/splitcondition.py CombinedCondition most_recent_split_condition")
             
        if len(self.conditions) == 0:
            print("-exit bartpy/bartpy/splitcondition.py CombinedCondition most_recent_split_condition")
            return None
        else:
            print("-exit bartpy/bartpy/splitcondition.py CombinedCondition most_recent_split_condition")
            return self.conditions[-1]
