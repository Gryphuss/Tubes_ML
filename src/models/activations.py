import numpy as np

# Base Class
class Activation:
    def activate(self, x):
        raise NotImplementedError
        
    def derivative(self, x):
        raise NotImplementedError
    
    def name(self):
        raise NotImplementedError


class Linear(Activation):
    def activate(self, x):
        return x
        
    def derivative(self, x):
        return np.ones_like(x)
    
    def name(self):
        return "Linear"


class ReLU(Activation):
    def activate(self, x):
        return np.maximum(0, x)
        
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def name(self):
        return "ReLU"


class Sigmoid(Activation):
    def activate(self, x):
        x_safe = np.clip(x, -709, 709)
        
        mask = x_safe < 0
        result = np.empty_like(x_safe)
        
        np.exp(x_safe[mask], out=result[mask])
        result[mask] = result[mask] / (1 + result[mask])
        
        result[~mask] = 1 / (1 + np.exp(-x_safe[~mask]))
        
        return result
        
    def derivative(self, x):
        s = self.activate(x)
        return s * (1 - s)
    
    def name(self):
        return "Sigmoid"


class Tanh(Activation):
    def activate(self, x):
        return np.tanh(x)
        
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def name(self):
        return "Tanh"


class Softmax(Activation):
    def activate(self, x):
        x_safe = x - np.max(x, axis=1, keepdims=True)
        
        exp_x = np.exp(x_safe)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        
        sum_exp_x = np.maximum(sum_exp_x, 1e-20)
        
        return exp_x / sum_exp_x
        
    def derivative(self, x):
        s = self.activate(x)
        
        n_samples = x.shape[0]
        n_classes = x.shape[1]
        jacobians = np.zeros((n_samples, n_classes, n_classes))
        
        for i in range(n_samples):
            for j in range(n_classes):
                for k in range(n_classes):
                    delta_jk = 1 if j == k else 0
                    jacobians[i, j, k] = s[i, j] * (delta_jk - s[i, k])
        
        return jacobians
    
    def name(self):
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
    return ACTIVATION_FUNCTIONS[activation_name]()