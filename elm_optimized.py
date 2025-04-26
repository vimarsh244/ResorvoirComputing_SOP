import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# ======================
# 1. Enhanced Data Generation
# ======================
def generate_logistic_map(p_values, num_points=2000, noise_std=0.005):
    """Generate higher-resolution time series with better noise control"""
    all_series = []
    for p in p_values:
        x = np.zeros(num_points)
        x[0] = np.random.uniform(0.1, 0.9)
        for t in range(num_points-1):
            # Improved noise injection with damping
            clean = p * x[t] * (1 - x[t])
            x[t+1] = 0.95*clean + 0.05*np.random.normal(0, noise_std)
        # Adaptive clipping
        x = np.clip(x, 0.001, 0.999)
        all_series.append(x)
    return np.array(all_series)

# Increase parameter resolution
p_values = np.linspace(3.4, 4.0, 100)  # Increased from 50 to 100
time_series = generate_logistic_map(p_values, num_points=2000)

# ======================
# 2. Optimized ELM Implementation
# ======================
class EnhancedELM(torch.nn.Module):
    def __init__(self, hidden_size=8):  # Increased hidden size
        super().__init__()
        self.hidden_size = hidden_size
        
        # Enhanced initialization
        self.input_layer = torch.nn.Linear(1, hidden_size)
        torch.nn.init.orthogonal_(self.input_layer.weight)
        self.activation = torch.nn.Tanh()
        
        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)
        
        # Partial freezing with trainable bias
        self.input_layer.weight.requires_grad_(False)
        self.input_layer.bias.requires_grad_(True)  # Trainable biases

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        return self.output_layer(x)

# ======================
# 3. Batch Training with Regularization
# ======================
def train_elm_batch(series_list, hidden_size=8, reg=1e-4):
    """Batch training with improved regularization"""
    weights = []
    elm = EnhancedELM(hidden_size)
    
    # Batch processing
    X_list = [torch.FloatTensor(s[:-1]).unsqueeze(1) for s in series_list]
    y_list = [torch.FloatTensor(s[1:]).unsqueeze(1) for s in series_list]
    
    with torch.no_grad():
        H_list = [elm.activation(elm.input_layer(x)) for x in X_list]
        H = torch.cat(H_list, dim=0)
        y = torch.cat(y_list, dim=0)
        
        # Regularized pseudoinverse using torch
        solution = torch.linalg.lstsq(
            H.T @ H + reg * torch.eye(hidden_size),
            H.T @ y
        ).solution
    
    # Store weights for each parameter
    for i, s in enumerate(series_list):
        weights.append(solution[:, i*len(s)//len(series_list)].numpy())
    
    return elm, np.array(weights)

# Train with higher capacity model
elm, output_weights = train_elm_batch(time_series, hidden_size=8)

# ======================
# 4. Enhanced PCA and Reconstruction
# ======================
pca = PCA(n_components=1)
pca.fit(output_weights)
param_range = np.linspace(pca.components_[0].min(), 
                         pca.components_[0].max(), 
                         400).reshape(-1, 1)  # Higher resolution

def enhanced_reconstruction(pca, param_range, steps=2000, burn_in=1000):
    """Improved reconstruction with warmup and batch processing"""
    weights = pca.inverse_transform(param_range)
    
    # Batch simulation
    virtual_elms = []
    for w in weights:
        elm = EnhancedELM(8)
        elm.output_layer.weight.data = torch.FloatTensor(w.reshape(1, -1))
        virtual_elms.append(elm)
    
    # Parallel simulation
    states = []
    for elm in virtual_elms:
        x = torch.FloatTensor([[0.5]])
        orbit = []
        for _ in range(burn_in + steps):
            x = elm(x)
            orbit.append(x.item())
        states.append(orbit[burn_in:])
    
    return np.array(states)

reconstructed = enhanced_reconstruction(pca, param_range)

# ======================
# 5. Advanced Visualization
# ======================
fig = plt.figure(figsize=(18, 12))

# Original Bifurcation with Poincaré
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
for i, p in enumerate(p_values):
    ax1.scatter([p]*len(time_series[i]), time_series[i], 
               s=0.1, c=plt.cm.viridis(i/len(p_values)), alpha=0.7)
ax1.set_title("Original Bifurcation Diagram with Parameter Coloring")
ax1.set_xlabel("Parameter p")
ax1.set_ylabel("x")

# Reconstructed Bifurcation
ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
pc_values = pca.inverse_transform(param_range)[:,0]
for i, pc in enumerate(pc_values):
    ax2.scatter([pc]*len(reconstructed[i]), reconstructed[i], 
               s=0.1, c=plt.cm.plasma(i/len(pc_values)), alpha=0.7)
ax2.set_title("Enhanced Reconstructed Bifurcation")
ax2.set_xlabel("Principal Component Value")

# Poincaré Sections
def plot_poincare(ax, data, title):
    """Enhanced Poincaré plot with KDE"""
    x = data[:-1]
    y = data[1:]
    
    # Calculate point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    ax.scatter(x, y, c=z, s=0.5, cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+1)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Original Poincaré
ax3 = plt.subplot2grid((3, 2), (2, 0))
selected_p = len(p_values) // 2
plot_poincare(ax3, time_series[selected_p], "Original Poincaré Section")

# Reconstructed Poincaré
ax4 = plt.subplot2grid((3, 2), (2, 1))
selected_pc = len(reconstructed) // 2
plot_poincare(ax4, reconstructed[selected_pc], "Reconstructed Poincaré Section")

plt.tight_layout()
plt.show()