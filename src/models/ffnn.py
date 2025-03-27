import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from .activations import get_activation
from .losses import get_loss
from .initializers import get_initializer
from .utils import (ensure_2d_y,plot_weight_distribution, plot_network_graph, 
                   batch_iterator, get_regularizer, RMSNorm)


class FFNN:
    def __init__(self, layer_sizes, activations, loss='mse', 
                     weight_initializer='uniform', weight_init_params=None,
                     regularizer=None, use_rms_norm=False):
        """
        Initialize the network
        
        Parameters:
        -----------
        layer_sizes : list of integers
            Number of neurons in each layer, including input and output layers
        
        activations : list of strings or activation objects
            Activation functions for each layer
            Must be one less than layer_sizes (no activation for input layer)
        
        loss : string or loss object
            Loss function for the network
        
        weight_initializer : string or initializer object
            Weight initialization method
        
        weight_init_params : dict or None
            Parameters for weight initializer
        
        regularizer : string or regularizer object or None
            Regularization method (L1 or L2)
        
        use_rms_norm : bool
            Whether to use RMS normalization
        """
        
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least 2 layers (input and output)")
    
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activations must be one less than number of layers")
        
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        
        self.activations = []
        for act in activations:
            if isinstance(act,str):
                self.activations.append(get_activation(act))
            else:
                self.activations.append(act)
        
        if isinstance(loss,str):
            self.loss = get_loss(loss)
        else:
            self.loss = loss
        
        if isinstance(weight_initializer,str):
            self.initializer = get_initializer(weight_initializer, **weight_init_params)
        else:
            self.initializer = weight_initializer
            
    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        
        for i in range(self.n_layers - 1):
            weight_shape = (self.layer_sizes[i],self.layer_sizes[i+1])
            self.weights.append(self.initializer.initialize(weight_shape))
            self.weight_gradients.append(np.zeros(weight_shape))
            
            bias_shape = (self.layer_size[i+1],)
            self.biases.append(np.zeros(bias_shape))
            self.bias_gradients.append(np.zeros)
            
    def forward(self, X):
        """
        Forward propagation
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        list of ndarrays
            Pre-activation and post-activation values for each layer
        """
        # Shape: (n_samples, n_features) array size self.n_layers-1
        self.layer_inputs = [X]
        
        # Shape: (n_samples, n_features)
        self.pre_activations = [] # Net
        
        # Shape: (n_samples, n_features) array size self.n_layers-1
        self.post_activations = [X] # Output
        
        for i in range(self.n_layers-1):
            net = np.dot(self.post_activations[-1],self.weights[i]) + self.biases[i]
            self.pre_activations.append(net)
            
            output = self.activations[i].activate(net)
            self.post_activations.append(output)
        
        return self.post_activations[-1]
            
    
    def backward(self, y_true):
        """
        Backward propagation
        
        Parameters:
        -----------
        y_true : ndarray of shape (n_samples, n_outputs)
            Target values
            
        Returns:
        --------
        float
            Loss value
        """
        y_pred = self.post_activations[-1]
        
        loss_value = self.loss.compute(y_true, y_pred)
        
        batch_size = y_true.shape[0]
        # Error term output layer, shape: (n_sample,output)
        delta = self.loss.derivative(y_true, y_pred)
        
        # Mulai dari hidden layer terakhir
        for i in range(self.n_layers - 2, -1, -1):
            # derivative loss w.r.t weight layer i = error term (layer i+1) * post_activation layer i 
            
            # post_activation transposed shape: (n_output, n_sample)
            # weight_gradients: derivative loss w.r.t weight layer i (∂L/∂W)
            self.weight_gradients[i] = np.dot(self.post_activations[i].T,delta) / batch_size
            self.bias_gradients[i] = np.mean(delta,axis=0)
            
            if i>0:
                delta = delta.dot(self.weights[i].T)
                delta = delta * self.activations[i-1].derivative(self.pre_activations[i-1])
        
        return loss_value
    
    def update_weights(self, learning_rate):
        """
        Update weights and biases using gradient descent
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        """
        for i in range(self.n_layers - 1):
            self.weights[i] -= learning_rate * self.weight_gradients[i]
            self.biases -= learning_rate * self.bias_gradients[i]
            
    
    def fit(self, X, y, batch_size=32, learning_rate=0.01, epochs=100, 
                verbose=1, validation_data=None):
        """
        Train the network
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        y : ndarray of shape (n_samples, n_outputs) or (n_samples,)
            Target values
            
        batch_size : int
            Size of mini-batches
            
        learning_rate : float
            Learning rate for gradient descent
            
        epochs : int
            Number of training epochs
            
        verbose : int
            Verbosity level (0=silent, 1=show progress)
            
        validation_data : tuple of (X_val, y_val) or None
            Validation data
            
        Returns:
        --------
        dict
            Training history
        """
        # convert y value to 2D for consistency
        y = ensure_2d_y(y,self.activations[-1].name())
        
        # Row size train X
        n_samples = X.shape[0]
        history = {
            'train_loss': [],
            'val_loss' : [] if validation_data is not None else None
        }
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            
            iterator = batch_iterator(X,y, batch_size)
            n_batches = int(np.ceil(n_samples/batch_size))
            
            for batch_idx, (X_batch, y_batch) in enumerate(iterator):
                # Print progress every 10%
                if verbose == 1 and batch_idx % max(1,n_batches // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{n_batches}")
                
                self.forward(X_batch)
                
                batch_loss = self.backward(y_batch)
                # Batch loss contribution relative to an entire epoch
                epoch_loss += batch_loss * X_batch.shape[0] / n_samples
                
                self.update_weights(learning_rate)
            
            history['train_loss'].append(epoch_loss)
            if validation_data is not None:
                X_val, y_val = validation_data
                # convert y value to 2D for consistency
                y_val = ensure_2d_y(y_val,self.activations[-1].name())
        
                # Just to keep track val loss, no need for backward
                y_pred = self.forward(X_val)
                val_loss = self.loss.compute(y_val, y_pred)
                history['val_loss'].append(val_loss)
                
                if verbose == 1:
                    print(f'Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - '
                          f'loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}')
            elif verbose == 1:
                print(f'Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - '
                      f'loss: {epoch_loss:.4f}')
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        ndarray
            Predictions
        """
        return self.forward(X)
            
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
            
        y : ndarray of shape (n_samples, n_outputs) or (n_samples,)
            Target values
            
        Returns:
        --------
        float
            Loss value
        """
        y = ensure_2d_y(y, self.activations[-1].name())
        
        y_pred = self.predict(X)
        val_loss = self.loss.compute(y, y_pred)
        return val_loss
    
    def save(self, file_path):
        """
        Save model to file
        
        Parameters:
        -----------
        file_path : str
            Path to save file
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'layer_sizes': self.layer_sizes,
                'weights': self.weights,
                'biases': self.biases,
                'activations': self.activations,
                'loss': self.loss,
            }, f)
    
    def load(cls, file_path):
        """
        Load model from file
        
        Parameters:
        -----------
        file_path : str
            Path to model file
            
        Returns:
        --------
        FFNN
            Loaded model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            model = cls(
                layer_sizes=model_data['layer_sizes'],
                activations=model_data['activations'],
                loss=model_data['loss']
            )
            
            model.weights = model_data['weights']
            model.biases = model_data['biases']
            
            return model
    
    # Plotting (extra)
    
    def plot_model(self):
        """
        Visualize the network architecture with weights
        """
        from .utils import plot_network_graph
        plot_network_graph(self.layer_sizes, self.weights, self.biases, 
                        title='Neural Network Architecture')
    
    def plot_weight(self, layers=None):
        """
        Plot weight distribution of specified layers
        
        Parameters:
        -----------
        layers : list of int or None
            Indices of layers to plot. If None, plot all layers.
        """
        if layers is None:
            layers = list(range(len(self.weights)))
        
        weights_to_plot = [self.weights[i] for i in layers]
        layer_names = [f'Layer {i+1}' for i in layers]
        
        from .utils import plot_weight_distribution
        plot_weight_distribution(weights_to_plot, 
                               title='Weight Distribution')
    
    def plot_gradient_distribution(self, layers=None):
        """
        Plot gradient distribution of specified layers
        
        Parameters:
        -----------
        layers : list of int or None
            Indices of layers to plot. If None, plot all layers.
        """
        if layers is None:
            layers = list(range(len(self.weight_gradients)))
        gradients_to_plot = [self.weight_gradients[i] for i in layers]
        layer_names = [f'Layer {i+1}' for i in layers]
        from .utils import plot_weight_distribution
        plot_weight_distribution(gradients_to_plot, 
                               title='Gradient Distribution')