from typing import Any, List

import numpy as np


class NormalScalarSampler():

    def __init__(self,
                 cache_size: int=1000):
        #print("enter bartpy/bartpy/samplers/scalar.py NormalScalarSampler __init__")
        self._cache_size = cache_size
        self._cache = []
        #print("-exit bartpy/bartpy/samplers/scalar.py NormalScalarSampler __init__")

    def sample(self):
        #print("enter bartpy/bartpy/samplers/scalar.py NormalScalarSampler sample")
        if len(self._cache) == 0:
            self.refresh_cache()
        output = self._cache.pop()
        #print("-exit bartpy/bartpy/samplers/scalar.py NormalScalarSampler sample")
        return output

    def refresh_cache(self):
        #print("enter bartpy/bartpy/samplers/scalar.py NormalScalarSampler refresh_cache")
        self._cache = list(np.random.normal(size=self._cache_size))
        #print("-exit bartpy/bartpy/samplers/scalar.py NormalScalarSampler refresh_cache")


class UniformScalarSampler():

    def __init__(self,
                 cache_size: int=1000):
        #print("enter bartpy/bartpy/samplers/scalar.py UniformScalarSampler __init__")
        self._cache_size = cache_size
        self._cache = []
        #print("-exit bartpy/bartpy/samplers/scalar.py UniformScalarSampler __init__")

    def sample(self):
        #print("enter bartpy/bartpy/samplers/scalar.py UniformScalarSampler sample")
        if len(self._cache) == 0:
            self.refresh_cache()
        output = self._cache.pop()
        #print("-exit bartpy/bartpy/samplers/scalar.py UniformScalarSampler sample")
        return output

    def refresh_cache(self):
        #print("enter bartpy/bartpy/samplers/scalar.py UniformScalarSampler refresh_cache")
        self._cache = list(np.random.uniform(size=self._cache_size))
        #print("-exit bartpy/bartpy/samplers/scalar.py UniformScalarSampler refresh_cache")


class DiscreteSampler():

    def __init__(self,
                 values: List[Any],
                 probas: List[float]=None,
                 cache_size: int=1000):
        #print("enter bartpy/bartpy/samplers/scalar.py DiscreteSampler __init__")
        self._values = values
        if probas is None:
            probas = [1.0 / len(values) for x in values]
        self._probas = probas
        self._cache_size = cache_size
        self._cache = []
        #print("-exit bartpy/bartpy/samplers/scalar.py DiscreteSampler __init__")

    def sample(self):
        #print("enter bartpy/bartpy/samplers/scalar.py DiscreteSampler sample")
        if len(self._cache) == 0:
            self.refresh_cache()
        output = self._cache.pop()
        #print("-exit bartpy/bartpy/samplers/scalar.py DiscreteSampler sample")
        return output

    def refresh_cache(self):
        #print("enter bartpy/bartpy/samplers/scalar.py DiscreteSampler refresh_cache")
        self._cache = list(np.random.choice(self._values, p=self._probas, size=self._cache_size))
        #print("-exit bartpy/bartpy/samplers/scalar.py DiscreteSampler refresh_cache")
        
