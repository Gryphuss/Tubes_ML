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
    if n_classes is None:
        n_classes = np.max(y) + 1
    
    n_samples = len(y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot


def train_test_split(X, y, test_size=0.2, random_state=None):
    return 0


def batch_iterator(X, y, batch_size=32, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]


def plot_training_history(history, title='Training History'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'train_accuracy' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_weight_distribution(weights, title='Weight Distribution'):
    n_layers = len(weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))
    
    if n_layers == 1:
        axes = [axes]
    
    for i, layer_weights in enumerate(weights):
        if isinstance(layer_weights, tuple):
            w = layer_weights[0].flatten()
            b = layer_weights[1].flatten()
            axes[i].hist(w, bins=30, alpha=0.7, label='Weights')
            axes[i].hist(b, bins=30, alpha=0.7, label='Biases')
        else:
            w = layer_weights.flatten()
            axes[i].hist(w, bins=30)
        
        axes[i].set_title(f'Layer {i+1}')
        axes[i].set_xlabel('Weight Value')
        axes[i].set_ylabel('Frequency')
        if isinstance(layer_weights, tuple):
            axes[i].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_network_graph(layer_sizes, weights, biases, gradients=None, title='Neural Network Graph'):
    # Neural Network Graph maker
    # Idenya gini
    
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
        self.grad_gamma = None
    
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