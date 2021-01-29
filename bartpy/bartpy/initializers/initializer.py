from typing import Generator

from bartpy.bartpy.tree import Tree


class Initializer(object):
    """
    The abstract interface for the tree initializers.

    Initializers are responsible for setting the starting values of the model, in particular:
      - structure of decision and leaf nodes
      - variables and values used in splits
      - values of leaf nodes

    Good initialization of trees helps speed up convergence of sampling

    Default behaviour is to leave trees uninitialized
    """

    def initialize_tree(self, tree: Tree) -> None:
        #print("enter bartpy/bartpy/initializers/initializer.py Initializer initialize_tree")
        
        pass
        #print("-exit bartpy/bartpy/initializers/initializer.py Initializer initialize_tree")

    def initialize_trees(self, trees: Generator[Tree, None, None]) -> None:
        #print("enter bartpy/bartpy/initializers/initializer.py Initializer initialize_trees")
        for tree in trees:
            self.initialize_tree(tree)
        #print("-exit bartpy/bartpy/initializers/initializer.py Initializer initialize_trees")
