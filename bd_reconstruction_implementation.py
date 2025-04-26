import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_logistic_map(p_values, num_points=2000, noise_std=0.001):
    """Generate noisy logistic map time series with value clipping"""
    all_series = []
    for p in p_values:
        x = np.zeros(num_points)
        x[0] = np.random.uniform(0.1, 0.9)
        for t in range(num_points-1):
            # x[t+1] = p * x[t] * (1 - x[t]) + np.random.normal(0, noise_std)
            x[t+1] = p * x[t] * (1 - x[t])
            # Clip values to prevent explosion
            x[t+1] = np.clip(x[t+1], 0.001, 0.999)
        x += np.random.normal(0, noise_std, num_points)
        x = np.clip(x, 0.001, 0.999)  # Clip observational noise
        all_series.append(x)
    return np.array(all_series)

class ELM(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input normalization layer
        self.normalize = torch.nn.LayerNorm(1)
        
        # Hidden layer with smaller initialization
        self.input_layer = torch.nn.Linear(1, hidden_size, bias=True)
        torch.nn.init.normal_(self.input_layer.weight, std=0.1)
        # self.activation = torch.nn.Tanh()
        self.activation = torch.nn.Sigmoid()

        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)
        
        for param in self.input_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x = self.normalize(x)  # Added normalization
        x = self.input_layer(x)
        x = self.activation(x)
        return self.output_layer(x)

def train_elm(series_list, hidden_size=12, reg=1e-3):  # Increased regularization
    """Train ELM with numerical stability checks"""
    elms = []
    weights = []
    
    for series in series_list:
        # Normalize input data
        series = (series - np.mean(series)) / np.std(series)
        
        X = torch.FloatTensor(series[:-1]).unsqueeze(1)
        y = torch.FloatTensor(series[1:]).unsqueeze(1)
        
        elm = ELM(hidden_size)
        # print(elm)
        
        with torch.no_grad():
            H = elm.activation(elm.input_layer(elm.normalize(X)))
        
        H_np = H.numpy()
        y_np = y.numpy()
        
        # Regularized solution with condition check
        HTH = H_np.T @ H_np
        reg_matrix = reg * np.eye(HTH.shape[0])
        try:
            W = np.linalg.solve(HTH + reg_matrix, H_np.T @ y_np)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(HTH + reg_matrix, H_np.T @ y_np, rcond=None)[0]
        
        # Check for invalid values
        if np.any(np.isnan(W)) or np.any(np.isinf(W)):
            raise ValueError("Invalid weights detected")
            
        elms.append(elm)
        weights.append(W.flatten())
        print("Epoch: ", len(elms))
        print("Loss: ", np.mean((H_np @ W - y_np)**2))  
    
    return elms, np.array(weights)

# Generate data and train
elm_hidden_size = 4
p_values = np.linspace(3.4, 4.0, 10)
time_series = generate_logistic_map(p_values)
elms, output_weights = train_elm(time_series, hidden_size=elm_hidden_size, reg=1e-2)

if np.any(np.isnan(output_weights)) or np.any(np.isinf(output_weights)):
    raise ValueError("PCA input contains invalid values")

pca = PCA(n_components=1)
pca.fit(output_weights)
principal_components = pca.transform(output_weights)


# Create parameter interpolation along principal component
param_range = np.linspace(principal_components.min(), 
                          principal_components.max(), 
                          200)



def reconstruct_bifurcation(elms, pca, param_range, steps=1000, burn_in=500, hidden_size=4):
    """Reconstruct bifurcation diagram using PCA components"""
    bifurcation = []
    
    param_range_2d = param_range.reshape(-1, 1)
    
    # Get weights from PCA components
    weights = pca.inverse_transform(param_range_2d)
    
    for weight in weights:
        # Create virtual ELM with interpolated weights
        virtual_elm = ELM(hidden_size)
        virtual_elm.output_layer.weight.data = torch.FloatTensor(weight.reshape(1, -1))
        
        # Iterate the map
        x = torch.FloatTensor([[0.5]])
        states = []
        for _ in range(steps + burn_in):
            x = virtual_elm(x)
            states.append(x.item())
        
        bifurcation.append(states[burn_in:])
    
    return np.array(bifurcation)


# Generate reconstructed bifurcation
param_range = np.linspace(principal_components.min(), 
                         principal_components.max(), 
                         200).reshape(-1, 1)  # Ensure 2D shape

reconstructed = reconstruct_bifurcation(elms, pca, param_range, hidden_size=elm_hidden_size)

# plt.figure(figsize=(15, 5))


# plt.subplot(131)
# for i, p in enumerate(p_values):
#     plt.scatter([p]*len(time_series[i]), time_series[i], s=0.1, c='blue')
# plt.title("Original Bifurcation Diagram")
# plt.xlabel("Parameter p")
# plt.ylabel("x")

# plt.subplot(132)
# for i, p in enumerate(p_values):
#     plt.scatter([p]*len(time_series[i]), time_series[i], s=0.1, c='red')
# plt.title("Noisy Training Data")
# plt.xlabel("Parameter p")

# # Reconstructed Diagram
# plt.subplot(133)
# for i, pc in enumerate(param_range.flatten()):
#     plt.scatter([pc]*len(reconstructed[i]), reconstructed[i], 
#                 s=0.1, 
#                 c='green',
#                 alpha=0.5)
# plt.title("Reconstructed Bifurcation Diagram")
# plt.xlabel("Principal Component Value")
# plt.ylabel("System State")


# # plt.show()


# plt.tight_layout()
# plt.show()

def plot_full_logistic_map():
    """Plot the classic logistic bifurcation diagram"""
    r_values = np.linspace(2.5, 4.0, 1000)  # Full bifurcation range
    iterations = 2000
    last = 200
    
    fig = plt.figure(figsize=(15, 10))
    
    # Classic bifurcation diagram
    ax1 = plt.subplot(221)
    ax1.set_title("Theoretical Logistic Map")
    ax1.set_xlabel("Parameter r")
    ax1.set_ylabel("x")
    
    for r in r_values:
        x = 1e-5 * np.ones(1)
        for _ in range(iterations):
            x = r * x * (1 - x)
        x = np.concatenate([x, r * x * (1 - x)])
        ax1.plot([r] * len(x), x, ',k', alpha=0.25)
    
    # Clean numerical simulation
    ax2 = plt.subplot(222)
    ax2.set_title("Numerical Simulation (Clean)")
    ax2.set_xlabel("Parameter p")
    clean_series = generate_logistic_map(p_values, noise_std=0)
    for i, p in enumerate(p_values):
        ax2.scatter([p]*len(clean_series[i]), clean_series[i], s=0.1, c='blue')
    
    # Noisy training data
    ax3 = plt.subplot(223)
    ax3.set_title("Noisy Training Data")
    ax3.set_xlabel("Parameter p")
    for i, p in enumerate(p_values):
        ax3.scatter([p]*len(time_series[i]), time_series[i], s=0.1, c='red')
    
    # Reconstructed diagram
    ax4 = plt.subplot(224)
    ax4.set_title("ELM Reconstructed Diagram")
    ax4.set_xlabel("Principal Component Value")
    for i, pc in enumerate(param_range.flatten()):
        ax4.scatter([pc]*len(reconstructed[i]), reconstructed[i], s=0.1, c='green', alpha=0.5)


    #     # Reconstructed diagram
    # ax4 = plt.subplot(224)
    # ax4.set_title("ELM Reconstructed Diagram")
    # ax4.set_xlabel("parameter")
    # for i, p in enumerate(p_values):
    #     ax4.scatter([p]*len(time_series[i]), time_series[i], s=0.1, c='green', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

plot_full_logistic_map()