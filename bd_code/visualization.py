import matplotlib.pyplot as plt
import numpy as np

def plot_bifurcation(ax, bd_data, title, xlabel, ylabel, marker='.', markersize=0.5, color='black'):
    """Plots a bifurcation diagram."""
    ax.plot(bd_data['params'], bd_data['attractor'], ls='', marker=marker, ms=markersize, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Optional: Set limits if needed
    # xmin, xmax = np.min(bd_data['params']), np.max(bd_data['params'])
    # ax.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))


def plot_path_locus(ax_path, ax_locus, original_params, gamma_vectors, title_prefix):
    """Plots the original bifurcation path and the reconstructed locus."""
    n_points = len(original_params)
    path_indices = np.arange(1, n_points + 1)

    # Assuming 1D parameters for now
    ax_path.plot(path_indices, original_params, marker='o', ls='-')
    ax_path.set_title(f"{title_prefix} Bifurcation Path")
    ax_path.set_xlabel("Path number")
    ax_path.set_ylabel("Original Parameter (p)")
    ax_path.grid(True)

    # Assuming 1D reconstructed space (gamma has one column)
    if gamma_vectors.shape[1] == 1:
        ax_locus.plot(path_indices, gamma_vectors[:, 0], marker='o', ls='-')
        ax_locus.set_title(f"{title_prefix} Bifurcation Locus")
        ax_locus.set_xlabel("Locus number")
        ax_locus.set_ylabel("Reconstructed Parameter (γ)")
        ax_locus.grid(True)
    else:
         ax_locus.set_title(f"{title_prefix} Bifurcation Locus (2D+)")
         ax_locus.text(0.5, 0.5, 'Plotting for >1D locus not implemented here', ha='center', va='center')


def plot_return_map(ax, series_orig, series_pred, title):
    """Plots return maps comparing original and predicted series."""
    # Use different markers/colors as in paper Fig 5
    ax.plot(series_orig[:-1], series_orig[1:], 'k.', markersize=2, label='Original (Simulated Circuit)') # '.' black
    if series_pred is not None:
        ax.plot(series_orig[:-1], series_pred[1:], 'rx', markersize=4, label='ELM Prediction') # 'x' red
    ax.set_title(title)
    ax.set_xlabel("x(t) or y(t)")
    ax.set_ylabel("x(t+1) or y(t+1)")
    ax.legend()
    ax.grid(True)


def plot_lyapunov_exponents(ax, params, le_values, title, xlabel, ylabel="Lyapunov Exponent (μ or ψ)", marker='.', markersize=1, color='blue'):
     """Plots estimated Lyapunov exponents vs. parameter."""
     ax.plot(params, le_values, ls='', marker=marker, ms=markersize, color=color)
     ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Zero line
     ax.set_title(title)
     ax.set_xlabel(xlabel)
     ax.set_ylabel(ylabel)
     # Optional: set limits
     # ax.set_ylim(min(np.min(le_values), -1), max(np.max(le_values), 0.5))


def plot_le_convergence(ax, steps, log_divergence, title):
    """Plots the log divergence curve for LE_ts estimation."""
    ax.plot(steps, log_divergence, marker='.', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel("Time Steps (k)")
    ax.set_ylabel("Mean Log Divergence <ln(dist(k))>")
    ax.grid(True)
    # Optionally plot the fitted line
    if not np.isnan(log_divergence).any():
        fit_end = max(5, len(steps) // 10)
        if fit_end > 1:
            coeffs = np.polyfit(steps[1:fit_end], log_divergence[1:fit_end], 1)
            fit_line = coeffs[1] + coeffs[0] * steps
            ax.plot(steps, fit_line, 'r--', label=f'Fit Slope (LE) ≈ {coeffs[0]:.3f}')
            ax.legend()