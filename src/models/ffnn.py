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

        # bonus regularizer
        self.regularizer = None
        if regularizer:
            if isinstance(regularizer, str):
                self.regularizer = get_regularizer(regularizer, 
                                        **(weight_init_params or {}))
            else:
                self.regularizer = regularizer
        
        # bonur normalization
        self.use_rms_norm = use_rms_norm
        self.normalizers = []
        if use_rms_norm:
            for i in range(self.n_layers - 1):
                self.normalizers.append(RMSNorm())
        
        # Supaya gak error none
        if weight_init_params is None:
            weight_init_params = {}
        
        if isinstance(weight_initializer,str):
            self.initializer = get_initializer(weight_initializer, **weight_init_params)
        else:
            self.initializer = weight_initializer
        
        self._initialize_weights()
            
    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        
        for i in range(self.n_layers - 1):
            weight_shape = (self.layer_sizes[i],self.layer_sizes[i+1])
            self.weights.append(self.initializer.initialize(weight_shape))
            self.weight_gradients.append(np.zeros(weight_shape))
            
            bias_shape = (self.layer_sizes[i+1],)
            self.biases.append(np.zeros(bias_shape))
            self.bias_gradients.append(np.zeros)

            # if rms norm
            if self.use_rms_norm:
                self.normalizers[i].initialize(bias_shape)

            
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
        # print("ni hao")
        # Shape: (n_samples, n_features) array size self.n_layers-1
        self.layer_inputs = [X]
        
        # Shape: (n_samples, n_features)
        self.pre_activations = [] # Net
        
        # Shape: (n_samples, n_features) array size self.n_layers-1
        self.post_activations = [X] # Output
        
        for i in range(self.n_layers-1):
            net = np.dot(self.post_activations[-1],self.weights[i]) + self.biases[i]
            self.pre_activations.append(net)

            if self.use_rms_norm:
                net = self.normalizers[i].forward(net)
            
            output = self.activations[i].activate(net)
            self.post_activations.append(output)
        
        # print("POST ACTIVATION: ")
        # for i in range(self.n_layers):
        #     print(f"Layer: {i}",self.post_activations[i].shape)
        # print("PRE ACTIVATION: ")
        # for i in range(self.n_layers-1):
        #     print(f"Layer: {i}",self.pre_activations[i].shape)
        return self.post_activations[-1]
            
    def _mult_activation_derivative(self, temp_delta, activation_input, activation):
        if activation.name() != "Softmax":
            derivatives = activation.derivative(activation_input)
            derivatives = np.clip(derivatives, -1e10, 1e10)
            temp_delta = np.clip(temp_delta, -1e10, 1e10)
            
            return temp_delta * derivatives
    
        jacobian = activation.derivative(activation_input)
        new_delta = np.zeros_like(temp_delta)
        
        for i in range(temp_delta.shape[0]):
            jacobian[i] = np.clip(jacobian[i], -1e10, 1e10)
            temp_delta_i = np.clip(temp_delta[i], -1e10, 1e10)
            
            new_delta[i] = np.dot(temp_delta_i, jacobian[i])
        
        return new_delta
    
    def backward(self, y_true):
        """
        Backward propagation with improved numerical stability
        
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
    
        if self.regularizer:
            reg_loss = self.regularizer.compute(self.weights)
            loss_value += reg_loss
        
        batch_size = y_true.shape[0]
        
        is_special_case = (self.loss.name() == "Categorial Cross-Entropy" and 
                          self.activations[-1].name() == "Softmax") or (
                          self.loss.name() == "Binary Cross-Entropy" and 
                          self.activations[-1].name() == "Sigmoid")
        
        if is_special_case:
            delta = y_pred - y_true
        else:
            loss_gradient = self.loss.derivative(y_true, y_pred)
            loss_gradient = np.clip(loss_gradient, -1e10, 1e10)
            delta = self._mult_activation_derivative(
                loss_gradient, self.pre_activations[-1], self.activations[-1])
        
        for i in range(self.n_layers - 2, -1, -1):
            if self.use_rms_norm:
                delta = self.normalizers[i].backward(delta)
            
            delta_clipped = np.clip(delta, -1e10, 1e10)
            self.weight_gradients[i] = np.dot(self.post_activations[i].T, delta_clipped)
            self.bias_gradients[i] = np.sum(delta_clipped, axis=0)
    
            if self.regularizer:
                reg_grad = self.regularizer.derivative([self.weights[i]])[0]
                self.weight_gradients[i] += reg_grad
            
            if i > 0:
                delta = np.dot(delta_clipped, self.weights[i].T)
                delta = np.clip(delta, -1e10, 1e10)
                delta = self._mult_activation_derivative(
                    delta, self.pre_activations[i-1], self.activations[i-1])
        
        return loss_value
    
    def clip_gradients(self, max_norm=1.0):
        """
        Clip gradients to prevent exploding gradients
        
        Parameters:
        -----------
        max_norm : float
            Maximum L2 norm of the gradients
        """
        total_norm_squared = 0
        for grad in self.weight_gradients:
            total_norm_squared += np.sum(np.square(grad))
        
        total_norm = np.sqrt(total_norm_squared)
        
        if total_norm > max_norm:
            clip_factor = max_norm / (total_norm + 1e-6)
            
            for i in range(len(self.weight_gradients)):
                self.weight_gradients[i] *= clip_factor
                self.bias_gradients[i] *= clip_factor

    def update_weights(self, learning_rate, gradient_clip=1.0):
        """
        Update weights and biases using gradient descent with clipping

        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        gradient_clip : float
            Maximum gradient norm
        """
        self.clip_gradients(max_norm=gradient_clip)

        for i in range(self.n_layers - 1):
            epsilon = 1e-8

            self.weights[i] -= learning_rate * self.weight_gradients[i]

            self.biases[i] -= learning_rate * self.bias_gradients[i]

            if self.use_rms_norm:
                self.normalizers[i].update(learning_rate)
            
    
    def fit(self, X, y, batch_size=32, learning_rate=0.01, epochs=100, 
         verbose=1, validation_data=None, gradient_clip=0.0,
         learning_rate_decay=1.0, early_stopping_patience=None):
        """
        Train the network with tqdm progress bar
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        y : ndarray of shape (n_samples, n_outputs) or (n_samples,)
            Target values
            
        batch_size : int
            Size of mini-batches
            
        learning_rate : float
            Initial learning rate for gradient descent
            
        epochs : int
            Number of training epochs
            
        verbose : int
            Verbosity level (0=silent, 1=show progress)
            
        validation_data : tuple of (X_val, y_val) or None
            Validation data
            
        gradient_clip : float
            Maximum gradient norm (0.0 = no clipping)
            
        learning_rate_decay : float
            Factor to multiply learning rate each epoch
            
        early_stopping_patience : int or None
            Number of epochs with no improvement before stopping
            
        Returns:
        --------
        dict
            Training history
        """
        from tqdm import tqdm
        
        y = ensure_2d_y(y, self.activations[-1].name())
        
        n_samples = X.shape[0]
        history = {
            'train_loss': [],
            'val_loss': [] if validation_data is not None else None
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        current_lr = learning_rate
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            
            iterator = batch_iterator(X, y, batch_size)
            n_batches = int(np.ceil(n_samples/batch_size))
            
            for batch_idx, (X_batch, y_batch) in enumerate(tqdm(iterator, total=n_batches, 
                                                          desc=f'Epoch {epoch+1}/{epochs}')):
                # Forward pass
                self.forward(X_batch)
                
                # Backward pass
                batch_loss = self.backward(y_batch)
                
                if np.isnan(batch_loss) or np.isinf(batch_loss):
                    if verbose:
                        print(f"Warning: Loss is {batch_loss} at epoch {epoch+1}, batch {batch_idx+1}")
                        print("Reducing learning rate and skipping batch...")
                    current_lr *= 0.5
                    continue
                    
                epoch_loss += batch_loss * X_batch.shape[0] / n_samples
                
                if gradient_clip > 0:
                    self.clip_gradients(max_norm=gradient_clip)
                    
                self.update_weights(current_lr)
            
            history['train_loss'].append(epoch_loss)
            
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val = ensure_2d_y(y_val, self.activations[-1].name())
        
                y_pred = self.forward(X_val)
                val_loss = self.loss.compute(y_val, y_pred)
                history['val_loss'].append(val_loss)
                
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                        
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - '
                          f'loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.6f}')
            elif verbose:
                print(f'Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - '
                      f'loss: {epoch_loss:.4f} - lr: {current_lr:.6f}')
            
            current_lr *= learning_rate_decay
        
        return history
    
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
        Save model to file with detailed debugging

        Parameters:
        -----------
        file_path : str
            Path to save file
        """
        try:
            print(f"Starting save process to {file_path}")

            print(f"layer_sizes: {self.layer_sizes}")

            print(f"Number of weight matrices: {len(getattr(self, 'weights', []))}")
            for i, w in enumerate(getattr(self, 'weights', [])):
                print(f"Weight matrix {i} shape: {w.shape}")

            print(f"Number of bias vectors: {len(getattr(self, 'biases', []))}")
            for i, b in enumerate(getattr(self, 'biases', [])):
                print(f"Bias vector {i} shape: {b.shape if hasattr(b, 'shape') else 'unknown'}")

            print(f"Number of activation functions: {len(getattr(self, 'activations', []))}")
            for i, act in enumerate(getattr(self, 'activations', [])):
                print(f"Activation {i}: {act.__class__.__name__ if hasattr(act, '__class__') else 'unknown'}")

            loss_name = getattr(self.loss, 'name', lambda: 'unknown')()
            print(f"Loss function: {loss_name}")

            if getattr(self, 'use_rms_norm', False):
                print(f"RMS norm enabled, normalizers: {len(getattr(self, 'normalizers', []))}")

            model_data = {
                'layer_sizes': self.layer_sizes,
                'weights': self.weights,
                'biases': self.biases,
            }

            if hasattr(self, 'loss'):
                model_data['loss'] = self.loss

            if hasattr(self, 'activations') and self.activations:
                activation_list = []
                for act in self.activations:
                    if hasattr(act, 'name'):
                        activation_list.append(act.name())
                    else:
                        activation_list.append(str(act))
                model_data['activations'] = activation_list

            if hasattr(self, 'regularizer') and self.regularizer:
                model_data['regularizer'] = self.regularizer

            if hasattr(self, 'use_rms_norm'):
                model_data['use_rms_norm'] = self.use_rms_norm

            print("Model data prepared successfully, attempting to save...")

            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)

            print("Model saved successfully!")

        except Exception as e:
            print(f"Error during save: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @classmethod
    def load(cls, file_path):
        """
        Load model from file with detailed debugging
        
        Parameters:
        -----------
        file_path : str
            Path to model file
            
        Returns:
        --------
        FFNN
            Loaded model
        """
        try:
            print(f"Starting load process from {file_path}")
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
    
            print(f"File loaded. Model data type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print("Model data is a dictionary, checking contents...")
                for key, value in model_data.items():
                    if isinstance(value, list):
                        print(f"{key}: list with {len(value)} items")
                    else:
                        print(f"{key}: {type(value)}")
            else:
                print(f"Warning: Model data is not a dictionary, but {type(model_data)}")
            
            layer_sizes = model_data.get('layer_sizes')
            print(f"Layer sizes: {layer_sizes}")
            
            print(f"Creating minimal model with {len(layer_sizes)} layers")
            
            default_activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
            default_loss = 'categorical_cross_entropy'
            
            model = cls(
                layer_sizes=layer_sizes,
                activations=default_activations,
                loss=default_loss
            )
            
            print("Basic model created. Setting saved parameters...")
    
            if 'weights' in model_data:
                print(f"Setting weights: {len(model_data['weights'])} matrices")
                model.weights = model_data['weights']
                
            if 'biases' in model_data:
                print(f"Setting biases: {len(model_data['biases'])} vectors")
                model.biases = model_data['biases']
                
            if 'loss' in model_data:
                print("Setting loss function")
                model.loss = model_data['loss']
                
            if 'activations' in model_data:
                print(f"Setting activations: {len(model_data['activations'])} functions")
                if all(isinstance(act, str) for act in model_data['activations']):
                    from src.models.activations import get_activation
                    activation_objects = [get_activation(act) for act in model_data['activations']]
                    model.activations = activation_objects
                else:
                    model.activations = model_data['activations']
                    
            if 'regularizer' in model_data:
                print("Setting regularizer")
                model.regularizer = model_data.get('regularizer')
                
            if 'use_rms_norm' in model_data:
                print(f"Setting RMS norm: {model_data['use_rms_norm']}")
                model.use_rms_norm = model_data.get('use_rms_norm')
    
            print("Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error during load: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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