import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class ELM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ELM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Hidden layer with random weights and biases (not trained)
        self.hidden = nn.Linear(input_size, hidden_size)
        
        # Initialize hidden layer weights and biases randomly
        nn.init.uniform_(self.hidden.weight, -1, 1)
        nn.init.uniform_(self.hidden.bias, -1, 1)
        
        # Output layer (weights will be trained)
        self.output = nn.Linear(hidden_size, output_size, bias=False)
        
        # Freeze hidden layer parameters
        for param in self.hidden.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Pass input through hidden layer with sigmoid activation
        hidden_output = torch.sigmoid(self.hidden(x))
        
        # Pass through output layer
        output = self.output(hidden_output)
        return output
    
    def train_model(self, X, y):
        """
        Train the ELM model using Moore-Penrose pseudoinverse
        """
        # Get hidden layer output
        with torch.no_grad():
            H = torch.sigmoid(self.hidden(X))
        
        # Calculate output weights using pseudoinverse
        H_pinv = torch.pinverse(H)
        beta = torch.matmul(H_pinv, y)
        
        # Set output weights
        with torch.no_grad():
            self.output.weight.copy_(beta.t())


def generate_logistic_map_data(parameter_values, length=1000, noise_level=0.001, initial_value=0.4):
    """Generate time-series data from the logistic map with noise."""
    datasets = []
    
    for p in parameter_values:
        x = initial_value
        time_series = []
        
        # Discard transients
        for _ in range(100):
            x = p * x * (1 - x)
        
        for _ in range(length):
            # Apply logistic map with dynamical noise
            x = p * x * (1 - x) + np.random.normal(0, noise_level)
            
            # Add observational noise
            time_series.append(x + np.random.normal(0, noise_level))
        
        datasets.append(np.array(time_series))
    
    return datasets

def generate_sine_map_data(parameter_values, length=1000, noise_level=0.001, initial_value=0.4):
    """Generate time-series data from the sine map with noise."""
    datasets = []
    c1 = 0.8  # Coefficients from the paper
    c2 = 1.25
    
    for p in parameter_values:
        x = initial_value
        time_series = []
        
        # Discard transients
        for _ in range(100):
            x = c1 * np.sin(c2 * p * x)
        
        for _ in range(length):
            # Apply sine map with dynamical noise
            x = c1 * np.sin(c2 * p * x) + np.random.normal(0, noise_level)
            
            # Add observational noise
            time_series.append(x + np.random.normal(0, noise_level))
        
        datasets.append(np.array(time_series))
    
    return datasets

def reconstruct_bifurcation_diagram(time_series_datasets, parameter_values, hidden_size=4, 
                                    num_points=1000, initial_value=0.4):
    """Reconstruct bifurcation diagram from time-series datasets."""
    # Step 1: Create and train ELM models for each dataset
    elms = []
    weight_vectors = []
    
    for dataset in time_series_datasets:
        # Create input-output pairs for time-series prediction
        X = torch.tensor(dataset[:-1].reshape(-1, 1), dtype=torch.float32)
        y = torch.tensor(dataset[1:].reshape(-1, 1), dtype=torch.float32)
        
        # Create and train ELM model
        elm = ELM(input_size=1, hidden_size=hidden_size, output_size=1)
        elm.train_model(X, y)
        elms.append(elm)
        weight_vectors.append(elm.output.weight.data.numpy().flatten())
    
    # Step 2: Perform PCA on weight vectors
    weight_vectors = np.array(weight_vectors)
    mean_vector = np.mean(weight_vectors, axis=0)
    deviation_vectors = weight_vectors - mean_vector
    
    # Compute covariance matrix
    cov_matrix = np.cov(deviation_vectors, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate contribution ratio
    total = sum(eigenvalues)
    contribution_ratio = [val/total for val in eigenvalues]
    cumulative_ratio = np.cumsum(contribution_ratio)
    
    # Determine dimensionality from contribution ratio (80% threshold)
    dimension = np.argmax(cumulative_ratio >= 0.8) + 1
    
    # Extract principal components
    principal_components = eigenvectors[:, :dimension]
    
    # Calculate estimated parameter vectors (bifurcation locus)
    estimated_param_vectors = []
    for dv in deviation_vectors:
        gamma = np.linalg.lstsq(principal_components, dv, rcond=None)[0]
        estimated_param_vectors.append(gamma)
    
    # Step 3: Generate reconstructed bifurcation diagram
    # Create parameter range for bifurcation diagram
    param_min = min(parameter_values)
    param_max = max(parameter_values)
    param_range = np.linspace(param_min, param_max, num_points)
    
    # Create mapping between original and estimated parameters
    param_mapping = list(zip(parameter_values, [v[0] for v in estimated_param_vectors]))
    param_mapping.sort(key=lambda x: x[0])
    
    # Generate reconstructed bifurcation diagram
    reconstructed_bd = []
    
    for orig_param in param_range:
        # Find nearest neighbors in parameter space
        left_idx = 0
        while left_idx < len(param_mapping) - 1 and param_mapping[left_idx + 1][0] <= orig_param:
            left_idx += 1
        
        if left_idx == len(param_mapping) - 1:
            left_idx = len(param_mapping) - 2
            
        # Interpolate estimated parameter
        p1, gamma1 = param_mapping[left_idx]
        p2, gamma2 = param_mapping[left_idx + 1]
        
        if p1 == p2:
            t = 0
        else:
            t = (orig_param - p1) / (p2 - p1)
        
        est_param = gamma1 + t * (gamma2 - gamma1)
        
        # Interpolate weight vector
        w1 = deviation_vectors[parameter_values.tolist().index(p1)]
        w2 = deviation_vectors[parameter_values.tolist().index(p2)]
        interp_vector = w1 + t * (w2 - w1)
        
        # Create ELM with interpolated weights
        elm = ELM(input_size=1, hidden_size=hidden_size, output_size=1)
        with torch.no_grad():
            elm.output.weight.copy_(torch.tensor(interp_vector.reshape(1, -1) + mean_vector, 
                                              dtype=torch.float32))
        
        # Generate time series using the model
        x = torch.tensor([[initial_value]], dtype=torch.float32)
        time_series = [initial_value]
        
        # Discard transients
        for _ in range(100):
            x = elm(x)
        
        # Generate points for bifurcation diagram
        for _ in range(200):
            x = elm(x)
            time_series.append(x.item())
        
        reconstructed_bd.append((orig_param, time_series[-100:]))
    
    return reconstructed_bd, (dimension, contribution_ratio, estimated_param_vectors)


def lyapunov_exponent_from_timeseries(time_series, epsilon=0.025, n_p=10, psi=100):
    """Estimate Lyapunov exponent from time-series data."""
    # Convert to numpy array if it's not
    time_series = np.array(time_series)
    
    # Check if the time series is periodic
    if np.allclose(time_series[:50], time_series[50:100], rtol=1e-5, atol=1e-5):
        return 0.0  # Return 0 for periodic time series
    
    lyapunov_estimates = []
    
    for t_start in range(0, len(time_series) - psi - 2, 10):
        # Select initial point
        p_target = time_series[t_start]
        
        # Find points in epsilon neighborhood
        distances = np.abs(time_series - p_target)
        indices = np.argsort(distances)
        
        # Exclude the initial point itself
        indices = indices[1:2*n_p+1]
        
        if len(indices) < 2*n_p:
            continue
        
        # Form pairs of points
        p_epsilon = time_series[indices]
        pairs1 = p_epsilon[:n_p]
        pairs2 = p_epsilon[n_p:2*n_p]
        
        # Calculate mean vector of distance between pairs
        d_epsilon_t = np.mean(np.abs(pairs1 - pairs2))
        
        if d_epsilon_t < 1e-10:
            continue
        
        # Calculate next-step distance
        next_indices = indices + 1
        next_indices = next_indices[next_indices < len(time_series)]
        if len(next_indices) < 2*n_p:
            continue
            
        pairs1_next = time_series[next_indices[:n_p]]
        pairs2_next = time_series[next_indices[n_p:2*n_p]]
        d_epsilon_t_next = np.mean(np.abs(pairs1_next - pairs2_next))
        
        # Estimate Lyapunov exponent
        if d_epsilon_t_next > 0:
            lyapunov_estimates.append(np.log(d_epsilon_t_next / d_epsilon_t))
    
    if not lyapunov_estimates:
        return 0.0
    
    return np.mean(lyapunov_estimates)

def lyapunov_exponent_from_derivative(model, x_range, num_iterations=1000, discard=100):
    """Estimate Lyapunov exponent using derivative of the model."""
    lyapunov_sum = 0.0
    count = 0
    
    for x_init in x_range:
        x = torch.tensor([[float(x_init)]], requires_grad=True, dtype=torch.float32)
        
        # Discard transients
        with torch.no_grad():
            for _ in range(discard):
                x = model(x)
            x = x.requires_grad_(True)
            
        # Compute Lyapunov exponent
        log_sum = 0.0
        
        for t in range(num_iterations):
            # Forward pass
            y = model(x)
            
            # Compute derivative
            y.backward(torch.ones_like(y), retain_graph=True)
            derivative = x.grad.item()
            
            # Update log sum
            log_sum += np.log(abs(derivative) + 1e-10)
            
            # Reset gradients and update x
            x.grad.zero_()
            with torch.no_grad():
                x = y.detach()
            x.requires_grad_(True)
        
        # Add to total
        lyapunov_sum += log_sum / num_iterations
        count += 1
    
    return lyapunov_sum / count if count > 0 else 0.0


def visualize_results(param_values, estimated_param_vectors, 
                     original_bd, noisy_bd, reconstructed_bd,
                     original_lyap, reconstructed_lyap, title):
    """Visualize results of bifurcation diagram reconstruction."""
    # Plot bifurcation path and locus
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot bifurcation path
    ax1.plot(range(1, len(param_values) + 1), param_values, 'o-')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Original parameter value')
    ax1.set_title('Bifurcation Path')
    
    # Plot bifurcation locus
    estimated_params = [p[0] for p in estimated_param_vectors]
    ax2.plot(range(1, len(estimated_params) + 1), estimated_params, 'o-')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Estimated parameter value')
    ax2.set_title('Bifurcation Locus')
    plt.tight_layout()
    plt.savefig(f'{title}_path_locus.png', dpi=300)
    plt.show()
    
    # Plot bifurcation diagrams
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot original bifurcation diagram
    for param, points in original_bd:
        ax1.plot([param] * len(points), points, 'k.', markersize=0.5)
    ax1.set_xlabel('Parameter value')
    ax1.set_ylabel('State value')
    ax1.set_title('Original Bifurcation Diagram')
    
    # Plot noisy bifurcation diagram
    for param, points in noisy_bd:
        ax2.plot([param] * len(points), points, 'k.', markersize=0.5)
    ax2.set_xlabel('Parameter value')
    ax2.set_ylabel('State value')
    ax2.set_title('Noisy Bifurcation Diagram')
    
    # Plot reconstructed bifurcation diagram
    for param, points in reconstructed_bd:
        ax3.plot([param] * len(points), points, 'k.', markersize=0.5)
    ax3.set_xlabel('Parameter value')
    ax3.set_ylabel('State value')
    ax3.set_title('Reconstructed Bifurcation Diagram')
    plt.tight_layout()
    plt.savefig(f'{title}_bifurcation_diagrams.png', dpi=300)
    plt.show()
    
    # Plot Lyapunov exponents
    param_range = [p for p, _ in original_bd]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot original Lyapunov exponents
    ax1.plot(param_range, original_lyap)
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_xlabel('Parameter value')
    ax1.set_ylabel('Lyapunov exponent')
    ax1.set_title('Original Lyapunov Exponents')
    
    # Plot reconstructed Lyapunov exponents
    ax2.plot(param_range, reconstructed_lyap)
    ax2.axhline(y=0, color='r', linestyle='-')
    ax2.set_xlabel('Parameter value')
    ax2.set_ylabel('Lyapunov exponent')
    ax2.set_title('Reconstructed Lyapunov Exponents')
    plt.tight_layout()
    plt.savefig(f'{title}_lyapunov_exponents.png', dpi=300)
    plt.show()


def main():
    """Run experiments for both logistic and sine maps."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Reconstructing bifurcation diagram for the logistic map...")
    
    # Generate parameter values for the logistic map as in the paper
    p_values_logistic = np.array([-0.15 * np.cos(2 * np.pi * (n-1) / 8) + 3.7 
                                 for n in range(1, 10)])
    
    # Generate time-series data from the logistic map
    logistic_datasets = generate_logistic_map_data(p_values_logistic, 
                                                  length=1000, 
                                                  noise_level=0.0001)
    
    # Reconstruct bifurcation diagram
    reconstructed_bd_logistic, bd_info_logistic = reconstruct_bifurcation_diagram(
        logistic_datasets, p_values_logistic, hidden_size=4)
    
    # Get reconstruction info
    dimension, contribution_ratio, estimated_param_vectors = bd_info_logistic
    
    # Generate original and noisy bifurcation diagrams for comparison
    param_range = np.linspace(min(p_values_logistic), max(p_values_logistic), 100)
    
    original_bd = []
    noisy_bd = []
    
    for p in param_range:
        # Generate original data
        x = 0.4
        points = []
        
        # Discard transients
        for _ in range(100):
            x = p * x * (1 - x)
        
        # Generate points for bifurcation diagram
        for _ in range(100):
            x = p * x * (1 - x)
            points.append(x)
        
        original_bd.append((p, points))
        
        # Generate noisy data
        x = 0.4
        noisy_points = []
        
        # Discard transients
        for _ in range(100):
            x = p * x * (1 - x) + np.random.normal(0, 0.001)
        
        # Generate points for bifurcation diagram
        for _ in range(100):
            x = p * x * (1 - x) + np.random.normal(0, 0.001)
            noisy_points.append(x + np.random.normal(0, 0.001))
        
        noisy_bd.append((p, noisy_points))
    
    # Calculate Lyapunov exponents
    original_lyap = []
    reconstructed_lyap = []
    
    for p, points in original_bd:
        # Calculate original Lyapunov exponent analytically
        log_deriv_sum = 0.0
        x = 0.4
        
        # Discard transients
        for _ in range(100):
            x = p * x * (1 - x)
        
        # Calculate Lyapunov exponent
        for _ in range(1000):
            derivative = p * (1 - 2 * x)
            log_deriv_sum += np.log(abs(derivative))
            x = p * x * (1 - x)
        
        original_lyap.append(log_deriv_sum / 1000)
    
    # Calculate reconstructed Lyapunov exponents
    for p, points in reconstructed_bd_logistic:
        reconstructed_lyap.append(lyapunov_exponent_from_timeseries(points))
        
    # Visualize results for logistic map
    visualize_results(p_values_logistic, estimated_param_vectors, 
                     original_bd, noisy_bd, reconstructed_bd_logistic,
                     original_lyap, reconstructed_lyap, "Logistic Map")
    
    # ====== SINE MAP EXPERIMENT ======
    print("\nReconstructing bifurcation diagram for the sine map...")
    
    # Generate parameter values for the sine map as in the paper
    p_values_sine = np.array([-0.1 * np.cos(2 * np.pi * (n-1) / 8) + 2.7 
                             for n in range(1, 10)])
    
    # Generate time-series data
    sine_datasets = generate_sine_map_data(p_values_sine, length=1000, noise_level=0.001)
    
    # Reconstruct bifurcation diagram
    reconstructed_bd_sine, bd_info_sine = reconstruct_bifurcation_diagram(
        sine_datasets, p_values_sine, hidden_size=4)
    
    # Generate comparison data and visualize results
    # (Similar to logistic map code but with sine map equations)
    #
    # Get reconstruction info
    dimension_sine, contribution_ratio_sine, estimated_param_vectors_sine = bd_info_sine

    # Generate original and noisy bifurcation diagrams for comparison
    param_range = np.linspace(min(p_values_sine), max(p_values_sine), 1000)
    c1 = 0.8  # Coefficients from the paper
    c2 = 1.25

    original_bd_sine = []
    noisy_bd_sine = []

    for p in param_range:
        # Generate original data
        x = 0.4
        points = []
        
        # Discard transients
        for _ in range(100):
            x = c1 * np.sin(c2 * p * x)
        
        # Generate points for bifurcation diagram
        for _ in range(100):
            x = c1 * np.sin(c2 * p * x)
            points.append(x)
        
        original_bd_sine.append((p, points))
        
        # Generate noisy data
        x = 0.4
        noisy_points = []
        
        # Discard transients
        for _ in range(100):
            x = c1 * np.sin(c2 * p * x) + np.random.normal(0, 0.001)
        
        # Generate points for bifurcation diagram
        for _ in range(100):
            x = c1 * np.sin(c2 * p * x) + np.random.normal(0, 0.001)
            noisy_points.append(x + np.random.normal(0, 0.001))
        
        noisy_bd_sine.append((p, noisy_points))

    # Calculate Lyapunov exponents
    original_lyap_sine = []
    reconstructed_lyap_sine = []

    for p, points in original_bd_sine:
        # Calculate original Lyapunov exponent analytically
        log_deriv_sum = 0.0
        x = 0.4
        
        # Discard transients
        for _ in range(100):
            x = c1 * np.sin(c2 * p * x)
        
        # Calculate Lyapunov exponent
        for _ in range(1000):
            derivative = c1 * c2 * p * np.cos(c2 * p * x)
            log_deriv_sum += np.log(abs(derivative))
            x = c1 * np.sin(c2 * p * x)
        
        original_lyap_sine.append(log_deriv_sum / 1000)

    # Calculate reconstructed Lyapunov exponents
    for p, points in reconstructed_bd_sine:
        reconstructed_lyap_sine.append(lyapunov_exponent_from_timeseries(points))
        
    # Visualize results for sine map
    visualize_results(p_values_sine, estimated_param_vectors_sine, 
                     original_bd_sine, noisy_bd_sine, reconstructed_bd_sine,
                     original_lyap_sine, reconstructed_lyap_sine, "Sine Map")

if __name__ == "__main__":
    main()
