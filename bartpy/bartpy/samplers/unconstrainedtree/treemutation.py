from typing import Optional

from bartpy.bartpy.model import Model, ModelCGM
from bartpy.bartpy.mutation import TreeMutation
from bartpy.bartpy.samplers.sampler import Sampler
from bartpy.bartpy.samplers.scalar import UniformScalarSampler
from bartpy.bartpy.samplers.treemutation import TreeMutationLikihoodRatio
from bartpy.bartpy.samplers.treemutation import TreeMutationProposer
from bartpy.bartpy.samplers.unconstrainedtree.likihoodratio import UniformTreeMutationLikihoodRatio
from bartpy.bartpy.samplers.unconstrainedtree.proposer import UniformMutationProposer
from bartpy.bartpy.tree import Tree, mutate


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
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler __init__")
        self.proposer = proposer
        self.likihood_ratio = likihood_ratio
        self._scalar_sampler = scalar_sampler
        print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler __init__")

    def sample(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample")
        proposal = self.proposer.propose(tree)
        ratio = self.likihood_ratio.log_probability_ratio(model, tree, proposal)
        
        if self._scalar_sampler.sample() < ratio:
            print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample")
            return proposal
        else:
            print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample")
            return None
    def sample_cgm_g(self, model: ModelCGM, tree: Tree) -> Optional[TreeMutation]:
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample_cgm_g")
        proposal = self.proposer.propose(tree)
        ratio = self.likihood_ratio.log_probability_ratio_cgm_g(model, tree, proposal)
        
        if self._scalar_sampler.sample() < ratio:
            print("###############################")
            print("#")
            print("# G TREE PROPOSAL ACCEPTED")
            print("#")
            print("###############################")
            print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample_cgm_g")
            return proposal
        else:
            print("###############################")
            print("#")
            print("# G TREE PROPOSAL REJECTED")
            print("#")
            print("###############################")
            print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample_cgm_g")
            return None
    
    def sample_cgm_h(self, model: ModelCGM, tree: Tree) -> Optional[TreeMutation]:
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample_cgm_h")
        proposal = self.proposer.propose(tree)
        ratio = self.likihood_ratio.log_probability_ratio_cgm_h(model, tree, proposal)
        
        if self._scalar_sampler.sample() < ratio:
            print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample_cgm_h")
            return proposal
        else:
            print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler sample_cgm_h")
            return None

    def step(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler step")
        mutation = self.sample(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
        print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler step")
        return mutation
    
    def step_cgm_g(self, model: ModelCGM, tree: Tree) -> Optional[TreeMutation]:
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler step_cgm_g")
        mutation = self.sample_cgm_g(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
        print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler step_cgm_g")
        return mutation
    
    def step_cgm_h(self, model: ModelCGM, tree: Tree) -> Optional[TreeMutation]:
        print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler step_cgm_h")
        mutation = self.sample_cgm_h(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
        print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler step_cgm_h")
        return mutation


def get_tree_sampler(p_grow: float,
                     p_prune: float) -> Sampler:
    print("enter /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler get_tree_sampler")
    proposer = UniformMutationProposer([p_grow, p_prune])
    likihood = UniformTreeMutationLikihoodRatio([p_grow, p_prune])
    output = UnconstrainedTreeMutationSampler(proposer, likihood)
    print("-exit /bartpy/bartpy/samplers/unconstrainedtree/treemutation.py UnconstrainedTreeMutationSampler get_tree_sampler")
    return output
