import numpy as np

class Loss:
    def compute(y_true, y_pred):
        raise NotImplementedError
        
    def derivative(y_true, y_pred):
        raise NotImplementedError
    
    def name():
        raise NotImplementedError


class MSE(Loss):
    def compute(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
        
    def derivative(y_true, y_pred):
        # Gradient descent, not negative
        n = y_true.shape[0]
        return 2 * (y_pred - y_true) / n
    
    def name():
        return "MSE"


class BinaryCrossEntropy(Loss):
    def compute(y_real, y_pred):
        # Clip prevent log(0
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
        
    def derivative(y_true, y_pred):
        # Clip prevent dbagi 0
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        n = y_true.shape[0]
        return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / n
    
    def name():
        return "Binary Cross-Entropy"


class CategoricalCrossEntropy(Loss):
    def compute(y_true, y_pred):
        # Clip prevent log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0)
        
        # Binary
        if len(y_true.shape) == 2:
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        # Classes
        else:
            n = y_true.shape[0]
            return -np.sum(np.log(y_pred[np.arange(n), y_true])) / n
        
    def derivative(y_true, y_pred):
        
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0)
        
        # Convert class ke binary if needed
        if len(y_true.shape) == 1:
            n_samples = y_true.shape[0]
            n_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((n_samples, n_classes))
            y_true_one_hot[np.arange(n_samples), y_true] = 1
            y_true = y_true_one_hot
        
        n = y_true.shape[0]
        return -y_true / (n * y_pred)
    
    def name():
        return "Categorical Cross-Entropy"


LOSS_FUNCTIONS = {
    'mse': MSE,
    'binary_cross_entropy': BinaryCrossEntropy,
    'categorical_cross_entropy': CategoricalCrossEntropy
}


def get_loss(loss_name):
    loss_name = loss_name.lower()
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{loss_name}' not implemented")
    return LOSS_FUNCTIONS[loss_name]