import numpy as np

from bartpy.bartpy.model import Model, ModelCGM
from bartpy.bartpy.node import LeafNode
from bartpy.bartpy.samplers.sampler import Sampler
from bartpy.bartpy.samplers.scalar import NormalScalarSampler


class LeafNodeSampler(Sampler):
    """
    Responsible for generating samples of the leaf node predictions
    Essentially just draws from a normal distribution with prior specified by model parameters

    Uses a cache of draws from a normal(0, 1) distribution to improve sampling performance
    """

    def __init__(self,
                 scalar_sampler=NormalScalarSampler(60000)):
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler __init__")
        self._scalar_sampler = scalar_sampler
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler __init__")

    def step(self, model: Model, node: LeafNode) -> float:
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler step")
        sampled_value = self.sample(model, node)
        node.set_value(sampled_value)
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler step")
        return sampled_value
    
    def step_cgm_g(self, model: ModelCGM, node: LeafNode) -> float:
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler step_cgm_g")
        sampled_value = self.sample_cgm_g(model, node)
        node.set_value(sampled_value)
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler step_cgm_g")
        return sampled_value
    
    def step_cgm_h(self, model: ModelCGM, node: LeafNode) -> float:
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler step_cgm_h")
        sampled_value = self.sample_cgm_h(model, node)
        node.set_value(sampled_value)
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler step_cgm_h")
        return sampled_value

    def sample(self, model: Model, node: LeafNode) -> float:
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler sample")
        prior_var = model.sigma_m ** 2
        n = node.data.X.n_obsv
        likihood_var = (model.sigma.current_value() ** 2) / n
        likihood_mean = node.data.y.summed_y() / n
        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        output = posterior_mean + (self._scalar_sampler.sample() * np.power(posterior_variance / model.n_trees, 0.5))
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler sample")
        return output

    def sample_cgm_g(self, model: ModelCGM, node: LeafNode) -> float:
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler sample_cgm_g")

        prior_var = model.sigma_m ** 2
        W = node.data.W.values # needs to apply mask
        p = node.data.p.values # needs to apply mask
        sigma_g_i = (W/p + (1-W)/(1-p))*model.sigma.current_value()
        
        one_over_sigma_g_i_sqrd = 1./(sigma_g_i**2)
        posterior_variance = (
            1./(
                (1./prior_var) + 
                np.sum( (~node.data.mask).astype(int) * one_over_sigma_g_i_sqrd)
            )
        )
        
        post_mean_numerator = np.sum( 
            (~node.data.mask).astype(int) * (node.data.y.values*one_over_sigma_g_i_sqrd)
        )
        posterior_mean = post_mean_numerator * posterior_variance

        output = posterior_mean + (self._scalar_sampler.sample() * np.power(posterior_variance / model.n_trees_g, 0.5))
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler sample_cgm_g")
        return output

    def sample_cgm_h(self, model: ModelCGM, node: LeafNode) -> float:
        #print("enter bartpy/bartpy/samplers/leafnode.py LeafNodeSampler sample_cgm_h")

        prior_var = model.sigma_m ** 2
        W = node.data.W.values # needs to apply mask
        p = node.data.p.values # needs to apply mask
        sigma_h_i = (1./(p*(1-p)))*model.sigma.current_value()
        
        one_over_sigma_h_i_sqrd = 1./(sigma_h_i**2)
        posterior_variance = 1./( (1/prior_var) + np.sum(((~node.data.mask).astype(int))*(one_over_sigma_h_i_sqrd)))
        
        post_mean_numerator = np.sum((~node.data.mask).astype(int)*(node.data.y.values*one_over_sigma_h_i_sqrd))
        posterior_mean = post_mean_numerator*posterior_variance

        output = posterior_mean + (self._scalar_sampler.sample() * np.power(posterior_variance / model.n_trees_h, 0.5))
        #print("-exit bartpy/bartpy/samplers/leafnode.py LeafNodeSampler sample_cgm_h")
        return output

# class VectorizedLeafNodeSampler(Sampler):

#     def step(self, model: Model, nodes: List[LeafNode]) -> float:
#         sampled_values = self.sample(model, nodes)
#         for (node, sample) in zip(nodes, sampled_values):
#             node.set_value(sample)
#         return sampled_values[0]

#     def sample(self, model: Model, nodes: List[LeafNode]) -> List[float]:
#         prior_var = model.sigma_m ** 2
#         n_s = []
#         sum_s = []
        

