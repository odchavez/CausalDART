from typing import Any, Callable

import numpy as np

from bartpy.bartpy.model import Model, deep_copy_model
from bartpy.bartpy.mutation import TreeMutation


class TraceLogger():

    def __init__(self,
                 f_tree_mutation_log: Callable[[TreeMutation], Any]=lambda x: x is not None,
                 f_model_log: Callable[[Model], Any]=lambda x: deep_copy_model(x),
                 f_in_sample_prediction_log: Callable[[np.ndarray], Any]=lambda x: x):
        print("enter bartpy/bartpy/trace.py TraceLogger __init__")
        self.f_tree_mutation_log = f_tree_mutation_log
        self.f_model_log = f_model_log
        self.f_in_sample_prediction_log = f_in_sample_prediction_log
        print("-exit bartpy/bartpy/trace.py TraceLogger __init__")

    def __getitem__(self, item: str):
        print("enter bartpy/bartpy/trace.py TraceLogger __getitem__")
        if item == "Tree":
            print("-exit bartpy/bartpy/trace.py TraceLogger __getitem__")
            return self.f_tree_mutation_log
        if item == "Model":
            print("-exit bartpy/bartpy/trace.py TraceLogger __getitem__")
            return self.f_model_log
        if item == "In Sample Prediction":
            print("-exit bartpy/bartpy/trace.py TraceLogger __getitem__")
            return self.f_in_sample_prediction_log
        if item in ["Node", "Sigma"]:
            print("-exit bartpy/bartpy/trace.py TraceLogger __getitem__")
            return lambda x: None
        else:
            raise KeyError("No method for key {}".format(item))
        print("-exit bartpy/bartpy/trace.py TraceLogger __getitem__")