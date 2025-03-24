import numpy as np

# Base Class
class Activation:
    def activate(x):
        raise NotImplementedError
        
    def derivative(x):
        raise NotImplementedError
    
    def name():
        raise NotImplementedError


class Linear(Activation):
    def activate(x):
        return x
        
    def derivative(x):
        return np.ones_like(x)
    
    def name():
        return "Linear"


class ReLU(Activation):
    def activate(x):
        return np.maximum(0, x)
        
    def derivative(x):
        return np.where(x > 0, 1, 0)
    
    def name():
        return "ReLU"


class Sigmoid(Activation):
    def activate(x):
        # PRevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
        
    def derivative(x):
        s = Sigmoid.activate(x)
        return s * (1 - s)
        
    
    def name():
        return "Sigmoid"


class Tanh(Activation):
    def activate(x):
        return np.tanh(x)
        # Equal (e^x - e^-x) / (e^x + e^-x)
        
    def derivative(x):
        return 1 - np.tanh(x) ** 2
        # Equal (2 / (e^x - e^-x))^2
    
    def name():
        return "Tanh"


class Softmax(Activation):
    def activate(x):
        # Minus Max to prevent overflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def derivative(x):
        s = Softmax.activate(x)
        
        n_samples = x.shape[0]
        n_classes = x.shape[1]
        jacobians = np.zeros((n_samples, n_classes, n_classes))
        
        for i in range(n_samples):
            for j in range(n_classes):
                for k in range(n_classes):
                    delta_jk = 1 if j == k else 0
                    jacobians[i, j, k] = s[i, j] * (delta_jk - s[i, k])
        
        return jacobians
    
    def name():
        return "Softmax"


ACTIVATION_FUNCTIONS = {
    'linear': Linear,
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax
}


def get_activation(activation_name):
    activation_name = activation_name.lower()
    if activation_name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Activation function '{activation_name}' not implemented (yet)")
    return ACTIVATION_FUNCTIONS[activation_name]