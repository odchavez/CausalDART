from typing import Tuple
from operator import gt, le

from sklearn.ensemble import GradientBoostingRegressor

from bartpy.bartpy.initializers.initializer import Initializer
from bartpy.bartpy.mutation import GrowMutation
from bartpy.bartpy.node import split_node, LeafNode
from bartpy.bartpy.splitcondition import SplitCondition
from bartpy.bartpy.tree import Tree, mutate


class SklearnTreeInitializer(Initializer):
    """
    Initialize tree structure and leaf node values by fitting a single Sklearn GBR tree

    Both tree structure and leaf node parameters are copied across
    """

    def __init__(self,
                 max_depth: int=4,
                 min_samples_split: int=2,
                 loss: str='ls'):
        print("enter bartpy/bartpy/initializers/sklearntreeinitializer.py SklearnTreeInitializer __init__")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        print("-exit bartpy/bartpy/initializers/sklearntreeinitializer.py SklearnTreeInitializer __init__")

    def initialize_tree(self,
                        tree: Tree) -> None:
        print("enter bartpy/bartpy/initializers/sklearntreeinitializer.py SklearnTreeInitializer initialize_tree")
        params = {
            'n_estimators': 1,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'learning_rate': 0.8,
            'loss': self.loss
        }
        print("initializing GradientBoostingRegressor......................WHY?")
        clf = GradientBoostingRegressor(**params)
        print("fitting GradientBoostingRegressor...........................WHY?")
        fit = clf.fit(tree.nodes[0].data.X.data, tree.nodes[0].data.y.data)
        sklearn_tree = fit.estimators_[0][0].tree_
        map_sklearn_tree_into_bartpy(tree, sklearn_tree)
        print("-exit bartpy/bartpy/initializers/sklearntreeinitializer.py SklearnTreeInitializer initialize_tree")


def map_sklearn_split_into_bartpy_split_conditions(sklearn_tree, index: int) -> Tuple[SplitCondition, SplitCondition]:
    """
    Convert how a split is stored in sklearn's gradient boosted trees library to the bartpy representation

    Parameters
    ----------
    sklearn_tree: The full tree object
    index: The index of the node in the tree object

    Returns
    -------

    """
    print("enter bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_split_into_bartpy_split_conditions")
    output = (
        SplitCondition(sklearn_tree.feature[index], sklearn_tree.threshold[index], le),
        SplitCondition(sklearn_tree.feature[index], sklearn_tree.threshold[index], gt)
    )
    print("-exit bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_split_into_bartpy_split_conditions")
    return output


def map_sklearn_tree_into_bartpy(bartpy_tree: Tree, sklearn_tree):
    print("enter bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_tree_into_bartpy")
    nodes = [None for x in sklearn_tree.children_left]
    nodes[0] = bartpy_tree.nodes[0]

    def search(index: int=0):
        print("enter bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_tree_into_bartpy search")
        left_child_index, right_child_index = sklearn_tree.children_left[index], sklearn_tree.children_right[index]

        if left_child_index == -1:  # Trees are binary splits, so only need to check left tree
            print("-exit bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_tree_into_bartpy search")
            return

        searched_node: LeafNode = nodes[index]

        split_conditions = map_sklearn_split_into_bartpy_split_conditions(sklearn_tree, index)
        decision_node = split_node(searched_node, split_conditions)

        left_child: LeafNode = decision_node.left_child
        right_child: LeafNode = decision_node.right_child
        left_child.set_value(sklearn_tree.value[left_child_index][0][0])
        right_child.set_value(sklearn_tree.value[right_child_index][0][0])

        mutation = GrowMutation(searched_node, decision_node)
        mutate(bartpy_tree, mutation)

        nodes[index] = decision_node
        nodes[left_child_index] = decision_node.left_child
        nodes[right_child_index] = decision_node.right_child

        search(left_child_index)
        search(right_child_index)
        print("exit bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_tree_into_bartpy search")

    search()
    print("-exit bartpy/bartpy/initializers/sklearntreeinitializer.py map_sklearn_tree_into_bartpy")

