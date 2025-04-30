import numpy as np
import matplotlib.pyplot as plt
from dynamical_systems import logistic_map, generate_time_series, generate_bifurcation_data
from elm import ELM
from reconstruction import train_elms_for_params, reconstruct_parameter_space, generate_reconstructed_bd
from lyapunov import lyapunov_exponent_ts, lyapunov_exponent_derivative, lyapunov_exponent_elm_derivative
from visualization import (plot_bifurcation, plot_path_locus, plot_return_map,
                           plot_lyapunov_exponents, plot_le_convergence)

# --- Simulation Parameters ---
MAP_FUNC = logistic_map
PARAM_NAME = 'p'
X0 = 0.5
N_STEPS_TRAIN = 1200 # Total steps for generating training data
N_TRANSIENT_TRAIN = 200 # Transient steps for training data
N_STEPS_BD = 500     # Steps per param value for plotting BDs
N_TRANSIENT_BD = 200  # Transient steps for plotting BDs
N_ATTRACTOR_POINTS = 100 # Points to plot per param value in BD
N_STEPS_LE = 2000     # Steps for averaging LE calculation
N_TRANSIENT_LE = 500   # Transient for LE calculation

# Noise level to simulate measurement/dynamical noise (set to 0 for clean)
# Paper Fig 4b suggests noise is present. Let's use a small value.
NOISE_LEVEL = 0.001 # Standard deviation of Gaussian noise
NOISE_TYPE = 'dynamical' # 'observational' or 'dynamical' (or 'both' if implemented)

# --- Reconstruction Parameters ---
ELM_HIDDEN_NEURONS = 4 # As mentioned in paper
PCA_COMPONENTS = 1     # For 1D map reconstruction

# --- Parameter Path (Eq 18) ---
P = 9 # Number of parameter points for training ELMs
n_vals = np.arange(1, P + 1)
p_train_values = -0.15 * np.cos(2 * np.pi * (n_vals - 1) / 8) + 3.7
param_train_list = [[p] for p in p_train_values] # List of lists for generic trainer

print(f"Training Parameter Values (p): {p_train_values}")

# --- 1. Train ELMs ---
# Generate time series with noise to simulate circuit data for training
elms, betas = train_elms_for_params(
    map_func=MAP_FUNC,
    param_values_list=param_train_list,
    param_indices=[0], # Index of 'p'
    other_params_base=[],
    elm_hidden_neurons=ELM_HIDDEN_NEURONS,
    x0=X0,
    n_steps=N_STEPS_TRAIN,
    n_transient=N_TRANSIENT_TRAIN,
    noise_level=NOISE_LEVEL,
    noise_type=NOISE_TYPE
)

# --- 2. Reconstruct Parameter Space ---
pca, beta_mean, gamma_vectors = reconstruct_parameter_space(betas, n_components=PCA_COMPONENTS)
print(f"Reconstructed Parameters (gamma): \n{gamma_vectors}")

# --- 3. Generate Reconstructed BD ---
# Create a smooth path in gamma space (interpolate between trained points)
gamma_min, gamma_max = np.min(gamma_vectors[:, 0]), np.max(gamma_vectors[:, 0])
n_recon_bd_points = 200 # Number of points for the reconstructed BD x-axis
gamma_path_reconstruction = np.linspace(gamma_min, gamma_max, n_recon_bd_points).reshape(-1, 1)

reconstructed_bd_data = generate_reconstructed_bd(
    elm_template=elms[0], # Use the first ELM as a template for structure
    pca=pca,
    beta_mean=beta_mean,
    gamma_path_points=gamma_path_reconstruction,
    x0=X0,
    n_steps_iter=N_STEPS_BD,
    n_transient_iter=N_TRANSIENT_BD,
    n_attractor_points=N_ATTRACTOR_POINTS
)

# --- 4. Generate Original and "Noisy" BDs for Comparison ---
p_bd_values_orig = np.linspace(3.5, 4.0, 200) # Range for original BD plot
original_bd_data_clean = generate_bifurcation_data(
    MAP_FUNC, p_bd_values_orig, param_index=0, other_params=[], x0=X0,
    n_steps=N_STEPS_BD, n_transient=N_TRANSIENT_BD, noise_level=0.0,
    n_attractor_points=N_ATTRACTOR_POINTS
)
# Generate with noise similar to training data ("simulated circuit")
original_bd_data_noisy = generate_bifurcation_data(
    MAP_FUNC, p_bd_values_orig, param_index=0, other_params=[], x0=X0,
    n_steps=N_STEPS_BD, n_transient=N_TRANSIENT_BD, noise_level=NOISE_LEVEL, noise_type=NOISE_TYPE,
    n_attractor_points=N_ATTRACTOR_POINTS
)

# --- 5. Calculate Lyapunov Exponents ---
le_params_plot = np.linspace(3.5, 4.0, 100) # Fewer points for LE calculation
le_orig_deriv = []
# le_noisy_ts = []

print("Calculating LE (Original - Derivative)...")
for p in le_params_plot:
    le = lyapunov_exponent_derivative(MAP_FUNC, [p], X0, N_STEPS_LE, N_TRANSIENT_LE)
    le_orig_deriv.append(le)

# print("Calculating LE (Noisy - Time Series)...")
# for p in le_params_plot:
#      # Generate noisy series for LE_ts calculation
#      noisy_series_le = generate_time_series(MAP_FUNC, [p], X0, N_STEPS_LE, N_TRANSIENT_LE, NOISE_LEVEL, NOISE_TYPE)
#      # Estimate LE (handle potential errors/warnings)
#      try:
#          # Reduce max_steps for faster calculation if needed
#          le, _, _ = lyapunov_exponent_ts(noisy_series_le, max_steps=N_STEPS_LE//10)
#          le_noisy_ts.append(le)
#      except Exception as e:
#          print(f"Warning: LE_ts calculation failed for p={p}: {e}")
#          le_noisy_ts.append(np.nan) # Append NaN on failure


# LE for Reconstructed BD
le_recon_params_plot = gamma_path_reconstruction[:, 0]
le_recon_deriv_elm = []
le_recon_ts = [] # Optional: LE_ts on ELM generated series

print("Calculating LE (Reconstructed - ELM Derivative)...")
for i, gamma in enumerate(gamma_path_reconstruction):
     delta_beta = pca.inverse_transform(gamma.reshape(1, -1)).flatten()
     current_beta_vector = beta_mean + delta_beta
     current_beta_matrix = current_beta_vector.reshape(elms[0].n_output, elms[0].n_hidden)
     le = lyapunov_exponent_elm_derivative(elms[0], current_beta_matrix, X0, N_STEPS_LE, N_TRANSIENT_LE)
     le_recon_deriv_elm.append(le)

     # Optional: LE_ts on reconstructed series
     # recon_series_le = elms[0].iterate(X0, N_STEPS_LE + N_TRANSIENT_LE, beta=current_beta_matrix)[N_TRANSIENT_LE:]
     # try:
     #     le_ts, _, _ = lyapunov_exponent_ts(recon_series_le, max_steps=N_STEPS_LE//10)
     #     le_recon_ts.append(le_ts)
     # except Exception as e:
     #     le_recon_ts.append(np.nan)


# --- 6. Visualization ---
plt.style.use('seaborn-v0_8-paper') # Use a style closer to publication plots

# Figure 3: Path and Locus
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))
plot_path_locus(ax3a, ax3b, p_train_values, gamma_vectors, "Logistic Map")
fig3.tight_layout()

# Figure 4: Bifurcation Diagrams
fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8), sharex='col', sharey=True)
ax4a, ax4b = axes4[0]
ax4c, ax4d = axes4[1]

plot_bifurcation(ax4a, original_bd_data_clean, title="(a) Original BD (Clean)", xlabel="", ylabel="x", color='black')
plot_bifurcation(ax4b, original_bd_data_noisy, title=f"(b) Original BD (Noise={NOISE_LEVEL}, {NOISE_TYPE})", xlabel="", ylabel="", color='black')
# Find corresponding p range for noisy BD plot based on gamma range
# This mapping is non-trivial, let's plot noisy BD vs original p as done in paper figure
# plot_bifurcation(ax4c, original_bd_data_noisy, title="(c) BD from 'Circuit' (Noisy Data)", xlabel="Parameter p", ylabel="x", color='black')
# Replot 4b as 4c for consistency with paper description
plot_bifurcation(ax4c, original_bd_data_noisy, title="(c) BD from 'Circuit' (Simulated)", xlabel="Original Parameter (p)", ylabel="x", color='black')

plot_bifurcation(ax4d, reconstructed_bd_data, title="(d) Reconstructed BD", xlabel="Reconstructed Parameter (γ)", ylabel="", color='black')

# Set x-axis labels correctly
ax4c.set_xlabel("Original Parameter (p)")
ax4d.set_xlabel("Reconstructed Parameter (γ)")

# Adjust limits if needed (match paper)
for ax in axes4.flat:
    ax.set_ylim(0, 1)
ax4a.set_xlim(3.5, 4.0)
ax4b.set_xlim(3.5, 4.0)
ax4c.set_xlim(3.5, 4.0)
# ax4d.set_xlim(gamma_min, gamma_max) # Auto scale is probably fine

fig4.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
fig4.suptitle("Logistic Map Bifurcation Diagrams", fontsize=14)


# Figure 5: Return Maps (Example for p=3.55 (n=1) and p=3.85 (n=5))
p_example1 = p_train_values[0] # n=1
gamma_example1 = gamma_vectors[0, 0]
elm_example1 = elms[0]
beta_example1 = betas[0]

p_example2 = p_train_values[4] # n=5
gamma_example2 = gamma_vectors[4, 0]
elm_example2 = elms[4]
beta_example2 = betas[4]


# Generate data for return maps
series_orig1 = generate_time_series(MAP_FUNC, [p_example1], X0, N_STEPS_TRAIN, N_TRANSIENT_TRAIN, NOISE_LEVEL, NOISE_TYPE)
series_pred1 = elm_example1.predict(series_orig1[:-1].reshape(-1, 1)).flatten()
# Ensure both series have the same length by using series_pred1 with series_orig1[:-1]
series_orig1 = series_orig1[:-1]

series_orig2 = generate_time_series(MAP_FUNC, [p_example2], X0, N_STEPS_TRAIN, N_TRANSIENT_TRAIN, NOISE_LEVEL, NOISE_TYPE)
series_pred2 = elm_example2.predict(series_orig2[:-1].reshape(-1, 1)).flatten()
# Ensure both series have the same length by using series_pred2 with series_orig2[:-1]
series_orig2 = series_orig2[:-1]


fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(10, 4.5))
plot_return_map(ax5a, series_orig1, series_pred1, f"Return Map (p={p_example1:.3f}, γ={gamma_example1:.1f})")
plot_return_map(ax5b, series_orig2, series_pred2, f"Return Map (p={p_example2:.3f}, γ={gamma_example2:.1f})")
# Add inset like Fig 6? Requires manual zoom/plot
# ax_inset = fig5.add_axes([0.7, 0.6, 0.15, 0.15]) # Example position [left, bottom, width, height]
# plot_return_map(ax_inset, series_orig2, series_pred2, "")
# ax_inset.set_xlim(0.8, 0.86); ax_inset.set_ylim(0.15, 0.35) # Match Fig 6 limits
# ax_inset.set_xlabel(""); ax_inset.set_ylabel(""); ax_inset.set_title("")
# ax_inset.legend().set_visible(False)

fig5.tight_layout()
fig5.suptitle("Logistic Map Return Plots", fontsize=14)


# Figure 7 & 8: Lyapunov Exponents
fig78, axes78 = plt.subplots(2, 2, figsize=(10, 8), sharex='col')
ax7a, ax7b = axes78[0] # Time series based LE
ax8a, ax8b = axes78[1] # Derivative based LE

# Plot LE_ts (Noisy/Circuit vs Reconstructed) - Fig 7
# plot_lyapunov_exponents(ax7a, le_params_plot, le_noisy_ts, title="(7a) LE from 'Circuit' (Time Series)", xlabel="", ylabel="LE (ψ)")
plot_lyapunov_exponents(ax7b, le_recon_params_plot, le_recon_ts if le_recon_ts else np.full_like(le_recon_params_plot, np.nan), title="(7b) LE Reconstructed (Time Series)", xlabel="", ylabel="") # Need to calc le_recon_ts if desired

# Plot LE_deriv (Original/MATLAB vs Reconstructed) - Fig 8
plot_lyapunov_exponents(ax8a, le_params_plot, le_orig_deriv, title="(8a) LE Original (Derivative)", xlabel="Original Parameter (p)", ylabel="LE (μ)")
plot_lyapunov_exponents(ax8b, le_recon_params_plot, le_recon_deriv_elm, title="(8b) LE Reconstructed (ELM Derivative)", xlabel="Reconstructed Parameter (γ)", ylabel="")

# Set common y-limits for LE plots?
# min_le = np.nanmin([np.nanmin(le_orig_deriv), np.nanmin(le_noisy_ts), np.nanmin(le_recon_deriv_elm)])
# max_le = np.nanmax([np.nanmax(le_orig_deriv), np.nanmax(le_noisy_ts), np.nanmax(le_recon_deriv_elm)])
# ylim_le = (min(min_le, -1.5) if not np.isnan(min_le) else -1.5, max(max_le, 0.5) if not np.isnan(max_le) else 0.5)

# for ax in axes78.flat: ax.set_ylim(ylim_le)
ax7a.set_xlim(3.5, 4.0)
ax8a.set_xlim(3.5, 4.0)
# ax7b.set_xlim(gamma_min, gamma_max) # Auto scale ok
# ax8b.set_xlim(gamma_min, gamma_max) # Auto scale ok


fig78.tight_layout(rect=[0, 0.03, 1, 0.95])
fig78.suptitle("Logistic Map Lyapunov Exponents", fontsize=14)


plt.show()