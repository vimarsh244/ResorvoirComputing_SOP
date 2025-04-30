import numpy as np
from scipy.special import expit # Sigmoid function (faster than 1/(1+exp(-x)))
# from numpy.linalg import pinv # Can use numpy's or sklearn's
# from sklearn.utils.extmath import pinvh as pinv # More stable pseudoinverse
# from scipy.linalg import pinvh as pinv # More stable pseudoinverse
from scipy.linalg import pinv # More stable pseudoinverse

class ELM:
    """Extreme Learning Machine for time-series prediction."""
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons):
        self.n_input = n_input_neurons
        self.n_hidden = n_hidden_neurons
        self.n_output = n_output_neurons

        # Initialize input weights and hidden biases randomly (and keep them fixed)
        # Weights often initialized in [-1, 1] or scaled by sqrt(1/n_input)
        self.input_weights = np.random.uniform(-1.0, 1.0, (self.n_hidden, self.n_input))
        self.biases = np.random.uniform(-1.0, 1.0, (self.n_hidden, 1)) # Use broadcasting later

        self.output_weights = None # Beta - to be computed during training

    def _sigmoid(self, x):
        # return 1.0 / (1.0 + np.exp(-x))
         return expit(x) # Numerically stable sigmoid

    def _hidden_layer_output(self, X):
        """Calculate the output of the hidden layer H."""
        # X shape: (n_input, n_samples)
        # input_weights shape: (n_hidden, n_input)
        # biases shape: (n_hidden, 1)
        linear_output = self.input_weights @ X + self.biases
        hidden_output = self._sigmoid(linear_output)
        # hidden_output shape: (n_hidden, n_samples)
        return hidden_output

    def train(self, X_train, y_train):
        """
        Train the ELM to find the output weights (beta).

        Args:
            X_train: Input data (time series at time t). Shape (n_input, n_samples).
                     For 1D map: (1, n_samples).
            y_train: Target data (time series at time t+1). Shape (n_output, n_samples).
                     For 1D map: (1, n_samples).
        """
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(1, -1)

        H = self._hidden_layer_output(X_train) # Shape (n_hidden, n_samples)
        # Calculate output weights beta = pinv(H.T) @ y_train.T
        # Or beta = y_train @ pinv(H) if shapes are (n_output, n_samples) and (n_hidden, n_samples)
        # Let's use the second form: beta shape (n_output, n_hidden)
        H_inv = pinv(H) # pinv expects (N, M) -> (M, N)
        self.output_weights = y_train @ H_inv # Shape (n_output, n_hidden)


    def predict(self, X_test):
        """Predict output for new input data."""
        if self.output_weights is None:
            raise ValueError("ELM must be trained before prediction.")
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1) # Ensure 2D for matrix multiplication

        H_test = self._hidden_layer_output(X_test) # Shape (n_hidden, n_samples)
        # Prediction y = beta @ H
        y_pred = self.output_weights @ H_test # Shape (n_output, n_samples)
        return y_pred

    def iterate(self, y0, n_steps, beta=None):
        """Iterate the ELM as a dynamical system from an initial condition y0."""
        if beta is None:
            if self.output_weights is None:
                 raise ValueError("ELM must be trained or beta provided for iteration.")
            beta_to_use = self.output_weights
        else:
            beta_to_use = beta # Use provided beta (e.g., from reconstruction)

        if not isinstance(y0, (np.ndarray, float, int)):
             raise TypeError("y0 must be a number or numpy array")
        if isinstance(y0, (float, int)):
             y0_arr = np.array([[y0]]) # Ensure shape (1, 1) for 1D case
        else:
             y0_arr = y0.reshape(self.n_input, 1) # Ensure shape (n_input, 1)

        y_iterated = np.zeros((self.n_output, n_steps))
        current_y = y0_arr

        for i in range(n_steps):
             # Calculate hidden output for current_y
             H_current = self._hidden_layer_output(current_y) # Shape (n_hidden, 1)
             # Calculate next y using provided/trained beta
             next_y = beta_to_use @ H_current # Shape (n_output, 1)
             y_iterated[:, i] = next_y.flatten()
             # Update current_y for next iteration (handle input/output matching)
             if self.n_input == self.n_output:
                 current_y = next_y
             else:
                 # Handle mismatch if necessary (e.g., use only first output if n_output > n_input)
                 # Assuming n_input=n_output=1 for 1D maps
                 current_y = next_y

        return y_iterated.flatten() # Return as 1D array for simplicity

    def get_output_weights(self):
        return self.output_weights.flatten() # Return as 1D vector

    def set_output_weights_from_vector(self, beta_vector):
         self.output_weights = beta_vector.reshape(self.n_output, self.n_hidden)