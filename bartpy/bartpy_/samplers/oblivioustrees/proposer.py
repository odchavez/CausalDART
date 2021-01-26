from operator import le, gt
from typing import Callable, List, Mapping, Optional, Tuple

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.mutation import TreeMutation, GrowMutation, PruneMutation
from bartpy.node import LeafNode, DecisionNode, split_node
from bartpy.samplers.scalar import DiscreteSampler
from bartpy.samplers.treemutation import TreeMutationProposer
from bartpy.split import SplitCondition
from bartpy.tree import Tree


def grow_mutations(tree: Tree) -> List[TreeMutation]:
    print("enter bartpy/bartpy/samplers/oblivioustrees/proposer.py grow_mutations")
    output = [GrowMutation(x, sample_split_node(x)) for x in tree.leaf_nodes]
    print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py grow_mutations")
    return output


def prune_mutations(tree: Tree) -> List[TreeMutation]:
    print("enter bartpy/bartpy/samplers/oblivioustrees/proposer.py prune_mutations")
    output = [PruneMutation(x, LeafNode(x.split, depth=x.depth)) for x in tree.prunable_decision_nodes]
    print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py prune_mutations")
    return output


class UniformMutationProposer(TreeMutationProposer):

    def __init__(self,
                 p_grow: float=0.5,
                 p_prune: float=0.5):
        print("enter bartpy/bartpy/samplers/oblivioustrees/proposer.py UniformMutationProposer __init__")
        self.method_sampler = DiscreteSampler([grow_mutations, prune_mutations],
                                              [p_grow, p_prune],
                                              cache_size=1000)
        print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py UniformMutationProposer __init__")

    def propose(self, tree: Tree) -> TreeMutation:
        print("enter bartpy/bartpy/samplers/oblivioustrees/proposer.py UniformMutationProposer propose")

        method = self.method_sampler.sample()
        try:
            output = method(tree)
            print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py UniformMutationProposer propose")
            return output
        except NoSplittableVariableException:
            output = self.propose(tree)
            print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py UniformMutationProposer propose")
            return output
        except NoPrunableNodeException:
            output = self.propose(tree)
            print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py UniformMutationProposer propose")
            return output


def sample_split_condition(node: LeafNode) -> Optional[Tuple[SplitCondition, SplitCondition]]:
    """
    Randomly sample a splitting rule for a particular leaf node
    Works based on two random draws

      - draw a node to split on based on multinomial distribution
      - draw an observation within that variable to split on

    Returns None if there isn't a possible non-degenerate split
    """
    print("enter bartpy/bartpy/samplers/oblivioustrees/proposer.py sample_split_condition")
    
    split_variable = node.data.X.random_splittable_variable()
    split_value = node.data.X.random_splittable_value(split_variable)
    if split_value is None:
        print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py sample_split_condition")
        return None
    output = SplitCondition(split_variable, split_value, le), SplitCondition(split_variable, split_value, gt)
    print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py sample_split_condition")
    return output


def sample_split_node(node: LeafNode) -> DecisionNode:
    """
    Split a leaf node into a decision node with two leaf children
    The variable and value to split on is determined by sampling from their respective distributions
    """
    print("enter bartpy/bartpy/samplers/oblivioustrees/proposer.py sample_split_node")
    
    if node.is_splittable():
        conditions = sample_split_condition(node)
        output = split_node(node, conditions)
        print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py sample_split_node")
        return output
    else:
        output = DecisionNode(node.split,
                            LeafNode(node.split, depth=node.depth + 1),
                            LeafNode(node.split, depth=node.depth + 1),
                            depth=node.depth)
        print("-exit bartpy/bartpy/samplers/oblivioustrees/proposer.py sample_split_node")
        return output
