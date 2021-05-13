from typing import List

import numpy as np

from bartpy.bartpy.model import Model, ModelCGM
from bartpy.bartpy.mutation import TreeMutation, GrowMutation, PruneMutation
from bartpy.bartpy.node import LeafNode, TreeNode
from bartpy.bartpy.samplers.treemutation import TreeMutationLikihoodRatio
from bartpy.bartpy.sigma import Sigma
from bartpy.bartpy.tree import Tree


def log_grow_ratio(combined_node: LeafNode, left_node: LeafNode, right_node: LeafNode, sigma: Sigma, sigma_mu: float):

    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)
    n = combined_node.data.X.n_obsv
    n_l = left_node.data.X.n_obsv
    n_r = right_node.data.X.n_obsv

    first_term = (var * (var + n * sigma_mu)) / ((var + n_l * var_mu) * (var + n_r * var_mu))
    first_term = np.log(np.sqrt(first_term))

    combined_y_sum = combined_node.data.y.summed_y()
    left_y_sum = left_node.data.y.summed_y()
    right_y_sum = right_node.data.y.summed_y()

    left_resp_contribution = np.square(left_y_sum) / (var + n_l * sigma_mu)
    right_resp_contribution = np.square(right_y_sum) / (var + n_r * sigma_mu)
    combined_resp_contribution = np.square(combined_y_sum) / (var + n * sigma_mu)

    resp_contribution = left_resp_contribution + right_resp_contribution - combined_resp_contribution
    output = first_term + ((var_mu / (2 * var)) * resp_contribution)

    return output


def log_grow_ratio_cgm_g(combined_node: LeafNode, left_node: LeafNode, right_node: LeafNode, sigma: Sigma, sigma_mu: float, mu_g: float):
    
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)

    W=combined_node.data.W.values
    p=combined_node.data.p.values
    
    sigma_g_i_sqr = var * ( W/(p**2) + (1-W)/((1-p)**2) )
    
    sum_sigma_g_i_sqr_left = np.sum( (~left_node.data.mask).astype(int) * 1./sigma_g_i_sqr)
    sum_sigma_g_i_sqr_right = np.sum( (~right_node.data.mask).astype(int) * 1./sigma_g_i_sqr)
    sum_sigma_g_i_sqr_combined = np.sum( (~combined_node.data.mask).astype(int) * 1./sigma_g_i_sqr)
    
    A_left = 1/var_mu + sum_sigma_g_i_sqr_left
    A_right = 1/var_mu + sum_sigma_g_i_sqr_right
    A_combined = 1/var_mu + sum_sigma_g_i_sqr_combined

    #first_term = .5 * (np.log(var_mu) + np.log(A_combined) - np.log(A_left) - np.log(A_right))
    first_term = (
        -.5 * (np.log(var_mu) + 
         .5 * np.log(A_combined) - 
         .5 * np.log(A_left) - 
         .5 * np.log(A_right))
    )
    
    y_tilde_g_i = combined_node.data.y.values
    y_tilde_g_i_over_var_i = y_tilde_g_i/sigma_g_i_sqr
    
    A_left_left_sum = (1/A_left)*(
        np.sum(
            (~left_node.data.mask).astype(int) * y_tilde_g_i_over_var_i 
        ) + mu_g/var_mu
    )**2
    A_right_right_sum = (1/A_right)*(
        np.sum(
            (~right_node.data.mask).astype(int) * y_tilde_g_i_over_var_i 
        ) + mu_g/var_mu
    )**2
    A_combined_combined_sum = (1/A_combined)*(
        np.sum(
            (~combined_node.data.mask).astype(int) * y_tilde_g_i_over_var_i
        ) + mu_g/var_mu
    )**2
    
    left_resp_contribution = 0.5 *  A_left_left_sum
    right_resp_contribution = 0.5 *  A_right_right_sum
    combined_resp_contribution = 0.5 *  A_combined_combined_sum
    
    resp_contribution = -.5*(mu_g**2)/var_mu + left_resp_contribution + right_resp_contribution - combined_resp_contribution

    output = first_term + resp_contribution

    return output


def log_grow_ratio_cgm_h(combined_node: LeafNode, left_node: LeafNode, right_node: LeafNode, sigma: Sigma, sigma_mu: float, mu_h: float):
    
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)

    W=combined_node.data.W.values
    p=combined_node.data.p.values
    
    sigma_h_i_sqr = var * ( W/(p**2) + (1-W)/((1-p)**2) )
    
    sum_sigma_h_i_sqr_left = np.sum( (~left_node.data.mask).astype(int) * 1./sigma_h_i_sqr)
    sum_sigma_h_i_sqr_right = np.sum( (~right_node.data.mask).astype(int) * 1./sigma_h_i_sqr)
    sum_sigma_h_i_sqr_combined = np.sum( (~combined_node.data.mask).astype(int) * 1./sigma_h_i_sqr)
    
    A_left = 1/var_mu + sum_sigma_h_i_sqr_left
    A_right = 1/var_mu + sum_sigma_h_i_sqr_right
    A_combined = 1/var_mu + sum_sigma_h_i_sqr_combined

    #first_term = .5 * (np.log(var_mu) + np.log(A_combined) - np.log(A_left) - np.log(A_right))
    first_term = (
        -.5 * (np.log(var_mu) + 
         .5 * np.log(A_combined) - 
         .5 * np.log(A_left) - 
         .5 * np.log(A_right))
    )
    
    y_tilde_h_i = combined_node.data.y.values
    y_tilde_h_i_over_var_i = y_tilde_h_i/sigma_h_i_sqr
    
    A_left_left_sum = (1/A_left)*(
        np.sum( 
            (~left_node.data.mask).astype(int) * y_tilde_h_i_over_var_i
        ) + mu_h/var_mu
    )**2
    A_right_right_sum = (1/A_right)*(
        np.sum(
            (~right_node.data.mask).astype(int) * y_tilde_h_i_over_var_i
        ) + mu_h/var_mu
    )**2
    A_combined_combined_sum = (1/A_combined)*(
        np.sum(
            (~combined_node.data.mask).astype(int) * y_tilde_h_i_over_var_i
        ) + mu_h/var_mu
    )**2
    
    left_resp_contribution = 0.5 *  A_left_left_sum
    right_resp_contribution = 0.5 *  A_right_right_sum
    combined_resp_contribution = 0.5 *  A_combined_combined_sum
    
    resp_contribution = -.5*(mu_h**2)/var_mu +  left_resp_contribution + right_resp_contribution - combined_resp_contribution

    output = first_term + resp_contribution

    return output

class UniformTreeMutationLikihoodRatio(TreeMutationLikihoodRatio):

    def __init__(self,
                 prob_method: List[float]=None):
        
        if prob_method is None:
            prob_method = [0.5, 0.5]
        self.prob_method = prob_method

    def log_transition_ratio(self, tree: Tree, mutation: TreeMutation):

        if mutation.kind == "prune":
            mutation: PruneMutation = mutation
            output = self.log_prune_transition_ratio(tree, mutation)
            return output
        if mutation.kind == "grow":
            mutation: GrowMutation = mutation
            output = self.log_grow_transition_ratio(tree, mutation)
            return output
        else:
            raise NotImplementedError("kind {} not supported".format(mutation.kind))

    def log_tree_ratio(self, model: Model, tree: Tree, mutation: TreeMutation):

        if mutation.kind == "grow":
            mutation: GrowMutation = mutation
            output = self.log_tree_ratio_grow(model, tree, mutation)
            return output
        if mutation.kind == "prune":
            mutation: PruneMutation = mutation
            output = self.log_tree_ratio_prune(model, mutation)
            return output

    def log_tree_ratio_cgm(self, model: ModelCGM, tree: Tree, mutation: TreeMutation):

        if mutation.kind == "grow":
            mutation: GrowMutation = mutation
            output = self.log_tree_ratio_grow_cgm(model, tree, mutation)
            return output
        if mutation.kind == "prune":
            mutation: PruneMutation = mutation
            output = self.log_tree_ratio_prune_cgm(model, mutation)
            return output

    def log_likihood_ratio(self, model: Model, tree: Tree, proposal: TreeMutation):

        if proposal.kind == "grow":
            proposal: GrowMutation = proposal
            output = self.log_likihood_ratio_grow(model, proposal)
            return output
        if proposal.kind == "prune":
            proposal: PruneMutation = proposal
            output = self.log_likihood_ratio_prune(model, proposal)
            return output
        else:
            raise NotImplementedError("Only prune and grow mutations supported")
        
    def log_likihood_ratio_cgm_g(self, model: ModelCGM, tree: Tree, proposal: TreeMutation):

        if proposal.kind == "grow":
            proposal: GrowMutation = proposal
            output = self.log_likihood_ratio_grow_cgm_g(model, proposal)
            return output
        if proposal.kind == "prune":
            proposal: PruneMutation = proposal
            output = self.log_likihood_ratio_prune_cgm_h(model, proposal)
            return output
        else:
            raise NotImplementedError("Only prune and grow mutations supported")

    def log_likihood_ratio_cgm_h(self, model: ModelCGM, tree: Tree, proposal: TreeMutation):

        if proposal.kind == "grow":
            proposal: GrowMutation = proposal
            output = self.log_likihood_ratio_grow_cgm_h(model, proposal)
            return output
        if proposal.kind == "prune":
            proposal: PruneMutation = proposal
            output = self.log_likihood_ratio_prune_cgm_h(model, proposal)
            return output
        else:
            raise NotImplementedError("Only prune and grow mutations supported")

    @staticmethod
    def log_likihood_ratio_grow(model: Model, proposal: TreeMutation):

        output = log_grow_ratio(
            proposal.existing_node, 
            proposal.updated_node.left_child, 
            proposal.updated_node.right_child, 
            model.sigma, model.sigma_m)
        
        return output

    @staticmethod
    def log_likihood_ratio_grow_cgm_g(model: ModelCGM, proposal: TreeMutation):

        output = log_grow_ratio_cgm_g(
            proposal.existing_node, 
            proposal.updated_node.left_child, 
            proposal.updated_node.right_child,
            model.sigma, model.sigma_g, model.mu_g)

        return output

    @staticmethod
    def log_likihood_ratio_grow_cgm_h(model: ModelCGM, proposal: TreeMutation):

        output = log_grow_ratio_cgm_h(
            proposal.existing_node, 
            proposal.updated_node.left_child, 
            proposal.updated_node.right_child, 
            model.sigma, model.sigma_h, model.mu_h)

        return output
    
    @staticmethod
    def log_likihood_ratio_prune(model: Model, proposal: TreeMutation):

        output = - log_grow_ratio(
            proposal.updated_node, 
            proposal.existing_node.left_child, 
            proposal.existing_node.right_child, 
            model.sigma, model.sigma_m)

        return output

    @staticmethod
    def log_likihood_ratio_prune_cgm_g(model: ModelCGM, proposal: TreeMutation):

        output = - log_grow_ratio_cgm_g(
            proposal.updated_node, 
            proposal.existing_node.left_child, 
            proposal.existing_node.right_child, 
            model.sigma, model.sigma_g, model.mu_g)

        return output

    @staticmethod
    def log_likihood_ratio_prune_cgm_h(model: ModelCGM, proposal: TreeMutation):

        output = - log_grow_ratio_cgm_h(
            proposal.updated_node, 
            proposal.existing_node.left_child, 
            proposal.existing_node.right_child, 
            model.sigma, model.sigma_h, model.mu_h)

        return output

    def log_grow_transition_ratio(self, tree: Tree, mutation: GrowMutation):

        prob_prune_selected = - np.log(n_prunable_decision_nodes(tree) + 1)
        prob_grow_selected = log_probability_split_within_tree(tree, mutation)

        prob_selection_ratio = prob_prune_selected - prob_grow_selected
        prune_grow_ratio = np.log(self.prob_method[1] / self.prob_method[0])
        output = prune_grow_ratio + prob_selection_ratio

        return output

    def log_prune_transition_ratio(self, tree: Tree, mutation: PruneMutation):

        if n_splittable_leaf_nodes(tree) == 1:
            prob_grow_node_selected = - np.inf  # Infinitely unlikely to be able to grow a null tree
        else:
            prob_grow_node_selected = - np.log(n_splittable_leaf_nodes(tree) - 1)
        prob_split = log_probability_split_within_node(GrowMutation(mutation.updated_node, mutation.existing_node))
        prob_grow_selected = prob_grow_node_selected + prob_split

        prob_prune_selected = - np.log(n_prunable_decision_nodes(tree))

        prob_selection_ratio = prob_grow_selected - prob_prune_selected
        grow_prune_ratio = np.log(self.prob_method[0] / self.prob_method[1])
        
        output = grow_prune_ratio + prob_selection_ratio

        return output

    @staticmethod
    def log_tree_ratio_grow(model: Model, tree: Tree, proposal: GrowMutation):

        denominator = log_probability_node_not_split(model, proposal.existing_node)

        prob_left_not_split = log_probability_node_not_split(model, proposal.updated_node.left_child)
        prob_right_not_split = log_probability_node_not_split(model, proposal.updated_node.right_child)
        prob_updated_node_split = log_probability_node_split(model, proposal.updated_node)
        prob_chosen_split = log_probability_split_within_tree(tree, proposal)
        numerator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split
        output = numerator - denominator

        return output
    
    @staticmethod
    def log_tree_ratio_grow_cgm(model: ModelCGM, tree: Tree, proposal: GrowMutation):

        denominator = log_probability_node_not_split(model, proposal.existing_node)

        prob_left_not_split = log_probability_node_not_split(model, proposal.updated_node.left_child)
        prob_right_not_split = log_probability_node_not_split(model, proposal.updated_node.right_child)
        prob_updated_node_split = log_probability_node_split(model, proposal.updated_node)
        prob_chosen_split = log_probability_split_within_tree(tree, proposal)
        numerator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split
        output = numerator - denominator

        return output

    @staticmethod
    def log_tree_ratio_prune(model: Model, proposal: PruneMutation):

        numerator = log_probability_node_not_split(model, proposal.updated_node)

        prob_left_not_split = log_probability_node_not_split(
            model, proposal.existing_node.left_child
        )
        prob_right_not_split = log_probability_node_not_split(
            model, proposal.existing_node.left_child
        )
        prob_updated_node_split = log_probability_node_split(
            model, proposal.existing_node
        )
        prob_chosen_split = log_probability_split_within_node(
            GrowMutation(proposal.updated_node, 
                         proposal.existing_node
                        )
        )
        denominator = (
            prob_left_not_split + 
            prob_right_not_split + 
            prob_updated_node_split + 
            prob_chosen_split
        )
        output = numerator - denominator

        return output

    @staticmethod
    def log_tree_ratio_prune_cgm(model: ModelCGM, proposal: PruneMutation):

        numerator = log_probability_node_not_split(model, proposal.updated_node)

        prob_left_not_split = log_probability_node_not_split(
            model, proposal.existing_node.left_child
        )
        prob_right_not_split = log_probability_node_not_split(
            model, proposal.existing_node.left_child
        )
        prob_updated_node_split = log_probability_node_split(
            model, proposal.existing_node
        )
        prob_chosen_split = log_probability_split_within_node(
            GrowMutation(proposal.updated_node, 
                         proposal.existing_node
                        )
        )
        denominator = (
            prob_left_not_split + 
            prob_right_not_split + 
            prob_updated_node_split + 
            prob_chosen_split
        )
        output = numerator - denominator

        return output

def n_prunable_decision_nodes(tree: Tree) -> int:
    """
    The number of prunable decision nodes
    i.e. how many decision nodes have two leaf children
    """
    output = len(tree.prunable_decision_nodes)
    return output


def n_splittable_leaf_nodes(tree: Tree) -> int:
    """
    The number of splittable leaf nodes
    i.e. how many leaf nodes have more than one distinct values in their covariate matrix
    """
    output = len(tree.splittable_leaf_nodes)
    return output


def log_probability_split_within_tree(tree: Tree, mutation: GrowMutation) -> float:
    """
    The log probability of the particular grow mutation being selected conditional on growing a given tree
    i.e.
    log(P(mutation | node)P(node| tree)

    """

    prob_node_chosen_to_split_on = - np.log(n_splittable_leaf_nodes(tree))
    prob_split_chosen = log_probability_split_within_node(mutation)
    output = prob_node_chosen_to_split_on + prob_split_chosen

    return output


def log_probability_split_within_node(mutation: GrowMutation) -> float:
    """
    The log probability of the particular grow mutation being selected conditional on growing a given node

    i.e.
    log(P(splitting_value | splitting_variable, node, grow) * P(splitting_variable | node, grow))
    """

    prob_splitting_variable_selected = - np.log(mutation.existing_node.data.X.n_splittable_variables)
    splitting_variable = mutation.updated_node.most_recent_split_condition().splitting_variable
    splitting_value = mutation.updated_node.most_recent_split_condition().splitting_value
    prob_value_selected_within_variable = np.log(
        mutation.existing_node.data.X.proportion_of_value_in_variable(
            splitting_variable, splitting_value
        )
    )
    output = prob_splitting_variable_selected + prob_value_selected_within_variable

    return output


def log_probability_node_split(model: Model, node: TreeNode):
    output = np.log(model.alpha * np.power(1 + node.depth, -model.beta))
    return output


def log_probability_node_not_split(model: Model, node: TreeNode):
    output = np.log(1. - model.alpha * np.power(1 + node.depth, -model.beta))
    return output

