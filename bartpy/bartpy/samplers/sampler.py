from abc import abstractmethod, ABC

from bartpy.bartpy.model import Model
from bartpy.bartpy.tree import Tree


class Sampler(ABC):

    @abstractmethod
    def step(self, model: Model, tree: Tree) -> bool:
        print("enter bartpy/bartpy/samplers/sampler.py Sampler step")
        raise NotImplementedError()
        print("-exit bartpy/bartpy/samplers/sampler.py Sampler step")
