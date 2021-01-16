from typing import Callable, Generator, Text, Tuple

import numpy as np
import pandas as pd

from bartpy.bartpy.model import Model, ModelCGM
from bartpy.bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.bartpy.samplers.sampler import Sampler
from bartpy.bartpy.samplers.sigma import SigmaSampler
from bartpy.bartpy.samplers.treemutation import TreeMutationSampler


class SampleSchedule:
    """
    The SampleSchedule class is responsible for handling the ordering of sampling within a Gibbs step
    It is useful to encapsulate this logic if we wish to expand the model

    Parameters
    ----------
    tree_sampler: TreeMutationSampler
        How to sample tree mutation space
    leaf_sampler: LeafNodeSampler
        How to sample leaf node predictions
    sigma_sampler: SigmaSampler
        How to sample sigma values
    """

    def __init__(self,
                 tree_sampler: TreeMutationSampler,
                 leaf_sampler: LeafNodeSampler,
                 sigma_sampler: SigmaSampler):
        print("enter bartpy/bartpy/samplers/schedule.py SampleSchedule __init__")
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler
        print("-exit bartpy/bartpy/samplers/schedule.py SampleSchedule __init__")

    def steps(self, model: Model) -> Generator[Tuple[Text, Callable[[], float]], None, None]:
        """
        Create a generator of the steps that need to be called to complete a full Gibbs sample

        Parameters
        ----------
        model: Model
            The model being sampled

        Returns
        -------
        Generator[Callable[[Model], Sampler], None, None]
            A generator a function to be called
        """
        print("enter bartpy/bartpy/samplers/schedule.py SampleSchedule steps")

        for tree in model.refreshed_trees():
            yield "Tree", lambda: self.tree_sampler.step(model, tree)
            for leaf_node in tree.leaf_nodes:
                yield "Node", lambda: self.leaf_sampler.step(model, leaf_node)
        yield "Node", lambda: self.sigma_sampler.step(model, model.sigma)
        print("-exit bartpy/bartpy/samplers/schedule.py SampleSchedule steps")


class SampleScheduleCGM:
    """
    The SampleSchedule class is responsible for handling the ordering of sampling within a Gibbs step
    It is useful to encapsulate this logic if we wish to expand the model

    Parameters
    ----------
    tree_sampler: TreeMutationSampler
        How to sample tree mutation space
    leaf_sampler: LeafNodeSampler
        How to sample leaf node predictions
    sigma_sampler: SigmaSampler
        How to sample sigma values
    """

    def __init__(self,
                 tree_sampler: TreeMutationSampler,
                 leaf_sampler: LeafNodeSampler,
                 sigma_sampler: SigmaSampler):
        print("enter bartpy/bartpy/samplers/schedule.py SampleScheduleCGM __init__")
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler
        print("-exit bartpy/bartpy/samplers/schedule.py SampleScheduleCGM __init__")

    def steps(self, model: Model) -> Generator[Tuple[Text, Callable[[], float]], None, None]:
        """
        Create a generator of the steps that need to be called to complete a full Gibbs sample

        Parameters
        ----------
        model: Model
            The model being sampled

        Returns
        -------
        Generator[Callable[[Model], Sampler], None, None]
            A generator a function to be called
        """
        print("enter bartpy/bartpy/samplers/schedule.py SampleScheduleCGM steps")
        # sample g and sigma_g
        for tree in model.refreshed_trees_g():
            yield "Tree", lambda: self.tree_sampler.step(model, tree)
            for leaf_node in tree.leaf_nodes:
                yield "Node", lambda: self.leaf_sampler.step(model, leaf_node)
        
        # sample h and sigma_h
        for tree in model.refreshed_trees_h():
            yield "Tree", lambda: self.tree_sampler.step(model, tree)
            for leaf_node in tree.leaf_nodes:
                yield "Node", lambda: self.leaf_sampler.step(model, leaf_node)
        yield "Node", lambda: self.sigma_sampler.step(model, model.sigma)
        print("-exit bartpy/bartpy/samplers/schedule.py SampleScheduleCGM steps")
