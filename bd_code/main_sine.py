import numpy as np
import matplotlib.pyplot as plt
from dynamical_systems import sine_map, generate_time_series, generate_bifurcation_data
from elm import ELM
from reconstruction import train_elms_for_params, reconstruct_parameter_space, generate_reconstructed_bd
from lyapunov import lyapunov_exponent_ts, lyapunov_exponent_derivative, lyapunov_exponent_elm_derivative
from visualization import (plot_bifurcation, plot_path_locus, plot_return_map,
                           plot_lyapunov_exponents, plot_le_convergence) # Add 2D BD plot later if needed

# --- Simulation Parameters ---
MAP_FUNC = sine_map
PARAM_NAME_1 = 'p1'
PARAM_NAME_2 = 'p2'
X0 = 0.5
N_STEPS_TRAIN = 1200
N_TRANSIENT_TRAIN = 200
N_STEPS_BD = 500
N_TRANSIENT_BD = 200
N_ATTRACTOR_POINTS = 100
N_STEPS_LE = 2000
N_TRANSIENT_LE = 500

NOISE_LEVEL = 0.001 # Similar noise level
NOISE_TYPE = 'dynamical'

# --- Reconstruction Parameters ---
ELM_HIDDEN_NEURONS = 4
PCA_COMPONENTS = 1 # For 1D sine map reconstruction

# --- Parameter Path (Eq 20) ---
P = 9
n_vals = np.arange(1, P + 1)
p1_train_values = -0.1 * np.cos(2 * np.pi * (n_vals - 1) / 8) + 2.7
p2_train_value = 0.0 # Fixed p2 for 1D reconstruction
# param_train_list needs both p1 and p2
param_train_list = [[p1, p2_train_value] for p1 in p1_train_values]

print(f"Training Parameter Values (p1): {p1_train_values}")

# --- 1. Train ELMs ---
elms, betas = train_elms_for_params(
    map_func=MAP_FUNC,
    param_values_list=param_train_list,
    param_indices=[0], # Index of p1 varied
    other_params_base=[], # Not needed as list contains all params
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
gamma_min, gamma_max = np.min(gamma_vectors[:, 0]), np.max(gamma_vectors[:, 0])
n_recon_bd_points = 200
gamma_path_reconstruction = np.linspace(gamma_min, gamma_max, n_recon_bd_points).reshape(-1, 1)

reconstructed_bd_data = generate_reconstructed_bd(
    elm_template=elms[0],
    pca=pca,
    beta_mean=beta_mean,
    gamma_path_points=gamma_path_reconstruction,
    x0=X0,
    n_steps_iter=N_STEPS_BD,
    n_transient_iter=N_TRANSIENT_BD,
    n_attractor_points=N_ATTRACTOR_POINTS
)

# --- 4. Generate Original and "Noisy" BDs for Comparison ---
p1_bd_values_orig = np.linspace(2.0, 3.0, 200) # Range for sine BD plot
original_bd_data_clean = generate_bifurcation_data(
    MAP_FUNC, p1_bd_values_orig, param_index=0, other_params=[p2_train_value], # Pass fixed p2
    x0=X0, n_steps=N_STEPS_BD, n_transient=N_TRANSIENT_BD, noise_level=0.0,
    n_attractor_points=N_ATTRACTOR_POINTS
)
original_bd_data_noisy = generate_bifurcation_data(
    MAP_FUNC, p1_bd_values_orig, param_index=0, other_params=[p2_train_value],
    x0=X0, n_steps=N_STEPS_BD, n_transient=N_TRANSIENT_BD, noise_level=NOISE_LEVEL, noise_type=NOISE_TYPE,
    n_attractor_points=N_ATTRACTOR_POINTS
)

# --- 5. Calculate Lyapunov Exponents ---
le_params_plot = np.linspace(2.0, 3.0, 100)
le_orig_deriv = []
le_noisy_ts = []

print("Calculating LE (Original - Derivative)...")
for p1 in le_params_plot:
    le = lyapunov_exponent_derivative(MAP_FUNC, [p1, p2_train_value], X0, N_STEPS_LE, N_TRANSIENT_LE)
    le_orig_deriv.append(le)

print("Calculating LE (Noisy - Time Series)...")
for p1 in le_params_plot:
     noisy_series_le = generate_time_series(MAP_FUNC, [p1, p2_train_value], X0, N_STEPS_LE, N_TRANSIENT_LE, NOISE_LEVEL, NOISE_TYPE)
     try:
         le, _, _ = lyapunov_exponent_ts(noisy_series_le, max_steps=N_STEPS_LE//10)
         le_noisy_ts.append(le)
     except Exception as e:
         le_noisy_ts.append(np.nan)

le_recon_params_plot = gamma_path_reconstruction[:, 0]
le_recon_deriv_elm = []
print("Calculating LE (Reconstructed - ELM Derivative)...")
for i, gamma in enumerate(gamma_path_reconstruction):
     delta_beta = pca.inverse_transform(gamma.reshape(1, -1)).flatten()
     current_beta_vector = beta_mean + delta_beta
     current_beta_matrix = current_beta_vector.reshape(elms[0].n_output, elms[0].n_hidden)
     le = lyapunov_exponent_elm_derivative(elms[0], current_beta_matrix, X0, N_STEPS_LE, N_TRANSIENT_LE)
     le_recon_deriv_elm.append(le)

# --- 6. Visualization ---
plt.style.use('seaborn-v0_8-paper')

# Figure 9: Path and Locus
fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(10, 4))
plot_path_locus(ax9a, ax9b, p1_train_values, gamma_vectors, "Sine Map")
ax9a.set_ylabel("Original Parameter (p1)")
fig9.tight_layout()


# Figure 10: Bifurcation Diagrams
fig10, (ax10a, ax10b, ax10c) = plt.subplots(3, 1, figsize=(6, 9), sharex=False, sharey=True) # Share y axis

plot_bifurcation(ax10a, original_bd_data_clean, title="(a) Original BD (Clean)", xlabel="", ylabel="x")
plot_bifurcation(ax10b, original_bd_data_noisy, title=f"(b) Original BD ('Circuit', Noise={NOISE_LEVEL})", xlabel="", ylabel="x")
plot_bifurcation(ax10c, reconstructed_bd_data, title="(c) Reconstructed BD", xlabel="Reconstructed Parameter (γ)", ylabel="x")

# Set x-axis labels and limits
ax10a.set_xlabel("Original Parameter (p1)")
ax10b.set_xlabel("Original Parameter (p1)")
ax10a.set_xlim(2.0, 3.0)
ax10b.set_xlim(2.0, 3.0)
# ax10c.set_xlim(gamma_min, gamma_max) # Auto scale usually fine

# Set y-axis limits (sine map range is roughly 0 to 0.8 with c1=0.8)
ylim_sine = (0, 0.85)
for ax in fig10.axes: ax.set_ylim(ylim_sine)

fig10.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
fig10.suptitle("Sine Map Bifurcation Diagrams", fontsize=14)


# Figure 11 & 12: Lyapunov Exponents
fig11_12, axes11_12 = plt.subplots(2, 2, figsize=(10, 8), sharex='col')
ax11a, ax11b = axes11_12[0] # Time series based LE (Fig 11)
ax12a, ax12b = axes11_12[1] # Derivative based LE (Fig 12)

# Plot LE_ts (Noisy/Circuit vs Reconstructed) - Fig 11
plot_lyapunov_exponents(ax11a, le_params_plot, le_noisy_ts, title="(11a) LE from 'Circuit' (Time Series)", xlabel="", ylabel="LE (ψ)")
# plot_lyapunov_exponents(ax11b, le_recon_params_plot, le_recon_ts, title="(11b) LE Reconstructed (Time Series)", xlabel="", ylabel="") # Calculate le_recon_ts if needed
ax11b.set_title("(11b) LE Reconstructed (TS - Not Implemented)")


# Plot LE_deriv (Original/MATLAB vs Reconstructed) - Fig 12
plot_lyapunov_exponents(ax12a, le_params_plot, le_orig_deriv, title="(12a) LE Original (Derivative)", xlabel="Original Parameter (p1)", ylabel="LE (μ)")
plot_lyapunov_exponents(ax12b, le_recon_params_plot, le_recon_deriv_elm, title="(12b) LE Reconstructed (ELM Derivative)", xlabel="Reconstructed Parameter (γ)", ylabel="")


# Set common y-limits for LE plots?
min_le_s = np.nanmin([np.nanmin(le_orig_deriv), np.nanmin(le_noisy_ts), np.nanmin(le_recon_deriv_elm)])
max_le_s = np.nanmax([np.nanmax(le_orig_deriv), np.nanmax(le_noisy_ts), np.nanmax(le_recon_deriv_elm)])
ylim_le_s = (min(min_le_s, -1.0) if not np.isnan(min_le_s) else -1.0, max(max_le_s, 1.0) if not np.isnan(max_le_s) else 1.0)

for ax in axes11_12.flat: ax.set_ylim(ylim_le_s)
ax11a.set_xlim(2.0, 3.0)
ax12a.set_xlim(2.0, 3.0)
# ax11b.set_xlim(gamma_min, gamma_max)
# ax12b.set_xlim(gamma_min, gamma_max)


fig11_12.tight_layout(rect=[0, 0.03, 1, 0.95])
fig11_12.suptitle("Sine Map Lyapunov Exponents", fontsize=14)


# Figure 13: 2D Bifurcation Diagram (Requires separate implementation)
# This would involve:
# 1. Defining p1, p2 values from Table I.
# 2. Training ELMs for each (p1, p2) pair.
# 3. Running PCA with n_components=2.
# 4. Generating a grid or path in the 2D gamma space.
# 5. For each gamma point, calculating beta, iterating ELM.
# 6. Estimating periodicity (e.g., using FFT or cycle detection) or LE.
# 7. Plotting using imshow or pcolormesh with colors based on periodicity/LE.
print("\nSkipping 2D Sine Map reconstruction (Fig 13) - Requires significant additions.")


plt.show()