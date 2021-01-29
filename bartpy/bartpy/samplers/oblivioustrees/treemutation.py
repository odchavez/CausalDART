from typing import Optional, List

import numpy as np

from bartpy.model import Model
from bartpy.mutation import TreeMutation
from bartpy.samplers.sampler import Sampler
from bartpy.samplers.scalar import UniformScalarSampler
from bartpy.samplers.treemutation import TreeMutationLikihoodRatio
from bartpy.samplers.treemutation import TreeMutationProposer
from bartpy.samplers.oblivioustrees.likihoodratio import UniformTreeMutationLikihoodRatio
from bartpy.samplers.oblivioustrees.proposer import UniformMutationProposer
from bartpy.tree import Tree, mutate


class UnconstrainedTreeMutationSampler(Sampler):
    """
    A sampler for tree mutation space.
    Responsible for producing samples of ways to mutate a tree within a model

    Works by combining a proposer and likihood evaluator into:
     - propose a mutation
     - assess likihood
     - accept if likihood higher than a uniform(0, 1) draw

    Parameters
    ----------
    proposer: TreeMutationProposer
    likihood_ratio: TreeMutationLikihoodRatio
    """

    def __init__(self,
                 proposer: TreeMutationProposer,
                 likihood_ratio: TreeMutationLikihoodRatio,
                 scalar_sampler=UniformScalarSampler()):
        #print("enter bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler __init__")
        self.proposer = proposer
        self.likihood_ratio = likihood_ratio
        self._scalar_sampler = scalar_sampler
        #print("-exit bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler __init__")

    def sample(self, model: Model, tree: Tree) -> Optional[List[TreeMutation]]:
        #print("enter bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler sample")
        
        proposals: List[TreeMutation] = self.proposer.propose(tree)
        ratio = np.sum([self.likihood_ratio.log_probability_ratio(model, tree, x) for x in proposals])
        if self._scalar_sampler.sample() < ratio:
            #print("-exit bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler sample")
            return proposals
        else:
            #print("-exit bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler sample")
            return None

    def step(self, model: Model, tree: Tree) -> Optional[List[TreeMutation]]:
        #print("enter bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler step")
        
        mutations = self.sample(model, tree)
        if mutations is not None:
            for mutation in mutations:
                mutate(tree, mutation)
        #print("-exit bartpy/bartpy/samplers/oblivioustrees/treemutation.py UnconstrainedTreeMutationSampler step")
        return mutations


def get_tree_sampler(p_grow: float,
                     p_prune: float):
    #print("enter bartpy/bartpy/samplers/oblivioustrees/treemutation.py get_tree_sampler")
    proposer = UniformMutationProposer(p_grow, p_prune)
    likihood = UniformTreeMutationLikihoodRatio([p_grow, p_prune])
    output = UnconstrainedTreeMutationSampler(proposer, likihood)
    #print("-exit bartpy/bartpy/samplers/oblivioustrees/treemutation.py get_tree_sampler")
    return output
