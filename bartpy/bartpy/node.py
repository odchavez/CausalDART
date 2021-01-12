from typing import Union, Tuple

from bartpy.bartpy.data import Data
from bartpy.bartpy.split import Split, SplitCondition


class TreeNode(object):
    """
    A representation of a node in the Tree
    Contains two main types of information:
        - Data relevant for the node
        - Links to children nodes
    """
    def __init__(self, split: Split, depth: int, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        print("enter bartpy/bartpy/node.py TreeNode __init__")
        
        self.depth = depth
        self._split = split
        self._left_child = left_child
        self._right_child = right_child
        print("-exit bartpy/bartpy/node.py TreeNode __init__")

    @property
    def data(self) -> Data:
        print("enter bartpy/bartpy/node.py TreeNode data")
        print("-exit bartpy/bartpy/node.py TreeNode data")
        return self._split.data

    @property
    def left_child(self) -> 'TreeNode':
        print("enter bartpy/bartpy/node.py TreeNode left_child")
        print("-exit bartpy/bartpy/node.py TreeNode left_child")
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        print("enter bartpy/bartpy/node.py TreeNode right_child")
        print("-exit bartpy/bartpy/node.py TreeNode right_child")
        return self._right_child

    @property
    def split(self):
        print("enter bartpy/bartpy/node.py TreeNode split")
        print("-exit bartpy/bartpy/node.py TreeNode split")
        return self._split

    def update_y(self, y):
        print("enter bartpy/bartpy/node.py TreeNode update_y")
        self.data.update_y(y)
        if self.left_child is not None:
            self.left_child.update_y(y)
            self.right_child.update_y(y)
        print("-exit bartpy/bartpy/node.py TreeNode update_y")
        


class LeafNode(TreeNode):
    """
    A representation of a leaf node in the tree
    In addition to the normal work of a `Node`, a `LeafNode` is responsible for:
        - Interacting with `Data`
        - Making predictions
    """

    def __init__(self, split: Split, depth=0, value=0.0):
        print("enter bartpy/bartpy/node.py LeafNode __init__")
        self._value = value
        super().__init__(split, depth, None, None)
        print("-exit bartpy/bartpy/node.py LeafNode __init__")

    def set_value(self, value: float) -> None:
        print("enter bartpy/bartpy/node.py LeafNode set_value")
        self._value = value
        print("-exit bartpy/bartpy/node.py LeafNode set_value")

    @property
    def current_value(self):
        print("enter bartpy/bartpy/node.py LeafNode current_value")
        print("-exit bartpy/bartpy/node.py LeafNode current_value")
        return self._value

    def predict(self) -> float:
        print("enter bartpy/bartpy/node.py LeafNode predict")
        print("-exit bartpy/bartpy/node.py LeafNode predict")
        return self.current_value

    def is_splittable(self) -> bool:
        print("enter bartpy/bartpy/node.py LeafNode is_splittable")
        output = self.data.X.is_at_least_one_splittable_variable()
        print("-exit bartpy/bartpy/node.py LeafNode is_splittable")
        return output


class DecisionNode(TreeNode):
    """
    A `DecisionNode` encapsulates internal node in the tree
    Unlike a `LeafNode`, it contains very little actual logic beyond tying the tree together
    """

    def __init__(self, split: Split, left_child_node: TreeNode, right_child_node: TreeNode, depth=0):
        print("enter bartpy/bartpy/node.py DecisionNode__init__")
        super().__init__(split, depth, left_child_node, right_child_node)
        print("-exit bartpy/bartpy/node.py DecisionNode__init__")

    def is_prunable(self) -> bool:
        print("enter bartpy/bartpy/node.py DecisionNode")
        output = type(self.left_child) == LeafNode and type(self.right_child) == LeafNode
        print("-exit bartpy/bartpy/node.py DecisionNode")
        return output

    def most_recent_split_condition(self) -> SplitCondition:
        print("enter bartpy/bartpy/node.py DecisionNode")
        output = self.left_child.split.most_recent_split_condition()
        print("-exit bartpy/bartpy/node.py DecisionNode")
        return output


def split_node(node: LeafNode, split_conditions: Tuple[SplitCondition, SplitCondition]) -> DecisionNode:
    """
    Converts a `LeafNode` into an internal `DecisionNode` by applying the split condition
    The left node contains all values for the splitting variable less than the splitting value
    """
    print("enter bartpy/bartpy/node.py split_node")
    left_split = node.split + split_conditions[0]
    split_conditions[1].carry_n_obsv = node.data.X.n_obsv - left_split.data.X.n_obsv
    split_conditions[1].carry_y_sum = node.data.y.summed_y() - left_split.data.y.summed_y()

    right_split = node.split + split_conditions[1]
    output = DecisionNode(node.split,
                        LeafNode(left_split, depth=node.depth + 1),
                        LeafNode(right_split, depth=node.depth + 1),
                        depth=node.depth)
    print("-exit bartpy/bartpy/node.py split_node")
    return output


def deep_copy_node(node: TreeNode):
    print("enter bartpy/bartpy/node.py deep_copy_node")
    if type(node) == LeafNode:
        node: LeafNode = node
        output = LeafNode(node.split.out_of_sample_conditioner(), value=node.current_value, depth=node.depth)
        print("-exit bartpy/bartpy/node.py deep_copy_node")
        return output
    elif type(node) == DecisionNode:
        node: DecisionNode = node
        output = DecisionNode(node.split.out_of_sample_conditioner(), node.left_child, node.right_child, depth=node.depth)
        print("-exit bartpy/bartpy/node.py deep_copy_node")
        return output
    else:
        raise TypeError("Unsupported node type")
    print("-exit bartpy/bartpy/node.py deep_copy_node")
