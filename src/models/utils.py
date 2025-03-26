import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pickle

def ensure_2d_y(y, activation_name):
    if len(y.shape) == 1:
        if activation_name == "Softmax":
            print("One hot encode y")
        else:
            # Reshape to 2D with one column
            return y.reshape(-1,1)

def one_hot_encode(y, n_classes=None):
    return 0


def train_test_split(X, y, test_size=0.2, random_state=None):
    return 0


def batch_iterator(X, y, batch_size=32, shuffle=True):
    return 0


def plot_training_history(history, title='Training History'):
    return 0


def plot_weight_distribution(weights, title='Weight Distribution'):
    return 0


def plot_network_graph(layer_sizes, weights, biases, gradients=None, title='Neural Network Graph'):
    # Neural Network Graph maker
    
    # Args:
    # - layer_sizes: list of integers, number of neurons in each layer
    # - weights: list of weight matrices
    # - biases: list of bias vectors
    # - gradients: optional, list of weight gradient matrices
    # - title: string, title of the plot
    return 0

# ------------------------- BONUS (belum 100% checked dan belum 100% benar) -------------------------------
class Regularizer:
    def __init__(self, lambda_param=0.01):
        self.lambda_param = lambda_param
    
    def compute(self, weights):
        raise NotImplementedError
    
    def derivative(self, weights):
        raise NotImplementedError
    
    def name(self):
        raise NotImplementedError


class L1Regularizer(Regularizer):
    def compute(self, weights):
        return self.lambda_param * np.sum([np.sum(np.abs(w)) for w in weights])
    
    def derivative(self, weights):
        return [self.lambda_param * np.sign(w) for w in weights]
    
    def name(self):
        return f"L1 Regularizer (lambda={self.lambda_param})"


class L2Regularizer(Regularizer):
    def compute(self, weights):
        return 0.5 * self.lambda_param * np.sum([np.sum(np.square(w)) for w in weights])
    
    def derivative(self, weights):
        return [self.lambda_param * w for w in weights]
    
    def name(self):
        return f"L2 Regularizer (lambda={self.lambda_param})"


REGULARIZERS = {
    'l1': L1Regularizer,
    'l2': L2Regularizer
}


def get_regularizer(regularizer_name, lambda_param=0.01):
    if regularizer_name is None:
        return None
    
    regularizer_name = regularizer_name.lower()
    if regularizer_name not in REGULARIZERS:
        raise ValueError(f"Regularizer '{regularizer_name}' not implemented")
    return REGULARIZERS[regularizer_name](lambda_param)


class RMSNorm:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.gamma = None
    
    def initialize(self, shape):
        self.gamma = np.ones(shape)
    
    def forward(self, x):
        self.x = x
        
        self.rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + self.eps)
        
        self.normalized = x / self.rms
        
        return self.gamma * self.normalized
    
    def backward(self, grad_output):
        self.grad_gamma = np.sum(grad_output * self.normalized, axis=0)
        
        n_features = self.x.shape[1]
        grad_input = grad_output * self.gamma / self.rms
        
        grad_rms = -np.sum(grad_output * self.gamma * self.x / (self.rms ** 2), axis=1, keepdims=True)
        grad_input += grad_rms * self.x / (n_features * self.rms)
        
        return grad_input
    
    def update(self, learning_rate):
        self.gamma -= learning_rate * self.grad_gamma