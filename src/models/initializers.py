import numpy as np

class Initializer:
    def initialize(self, shape):
        raise NotImplementedError
        
    def name(self):
        raise NotImplementedError


class ZeroInitializer(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)
        
    def name(self):
        return "Zero Initializer"


class UniformInitializer(Initializer):
    def __init__(self, low=-0.05, high=0.05, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def initialize(self, shape):
        return self.rng.uniform(self.low, self.high, shape)
        
    def name(self):
        return f"Uniform Initializer (low={self.low}, high={self.high})"


class NormalInitializer(Initializer):
    def __init__(self, mean=0.0, std=0.05, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def initialize(self, shape):
        return self.rng.normal(self.mean, self.std, shape)
        
    def name(self):
        return f"Normal Initializer (mean={self.mean}, std={self.std})"


class XavierInitializer(Initializer):
    # KATANYA GINI
    
    # Weights are initialized with values drawn from a distribution with zero mean
    # and variance 2/(fan_in + fan_out), where fan_in is the number of input units
    # and fan_out is the number of output units.
    
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def initialize(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, shape)
        
    def name(self):
        return "Xavier (Glorot) Initializer"


class HeInitializer(Initializer):
    # KATANYA GINI
    
    # Weights are initialized with values drawn from a normal distribution with 
    # zero mean and variance 2/fan_in.
    
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def initialize(self, shape):
        fan_in, fan_out = shape
        std = np.sqrt(2 / fan_in)
        return self.rng.normal(0, std, shape)
        
    def name(self):
        return "He Initializer"


INITIALIZERS = {
    'zero': ZeroInitializer,
    'uniform': UniformInitializer,
    'normal': NormalInitializer,
    'xavier': XavierInitializer,
    'he': HeInitializer
}


def get_initializer(initializer_name, **kwargs):
    initializer_name = initializer_name.lower()
    if initializer_name not in INITIALIZERS:
        raise ValueError(f"Initializer '{initializer_name}' not implemented")
    return INITIALIZERS[initializer_name](**kwargs)