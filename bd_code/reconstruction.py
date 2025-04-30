import numpy as np
from sklearn.decomposition import PCA
from elm import ELM
from dynamical_systems import generate_time_series

def train_elms_for_params(map_func, param_values_list, param_indices, other_params_base,
                          elm_hidden_neurons, x0, n_steps, n_transient, noise_level=0.0, noise_type='observational'):
    """
    Trains one ELM for each parameter setting in the list.

    Args:
        map_func: The map function.
        param_values_list: List of parameter arrays/tuples, one for each ELM.
                           e.g., for logistic: [[3.55], [3.6], ...]
                           e.g., for 2D sine: [[2.7, 0.0], [2.7, 0.05], ...]
        param_indices: Indices of the parameters being varied (not strictly needed if list is full params).
        other_params_base: Base values for parameters NOT in param_values_list (not needed if list is full).
        elm_hidden_neurons: Number of hidden neurons for ELM.
        x0, n_steps, n_transient: Params for time series generation.
        noise_level, noise_type: Noise params for time series generation.

    Returns:
        List of trained ELM instances.
        List of the corresponding output weight vectors (beta).
    """
    elms = []
    betas = []
    n_params = len(param_values_list)

    # Assume 1D input/output for ELM based on typical map usage
    n_input = 1
    n_output = 1

    for i, params in enumerate(param_values_list):
        print(f"Training ELM {i+1}/{n_params} for params: {params}")
        # Generate time series for this parameter set
        series = generate_time_series(map_func, params, x0, n_steps, n_transient, noise_level, noise_type)

        # Prepare data for ELM (input = series[:-1], output = series[1:])
        X_train = series[:-1].reshape(1, -1)
        y_train = series[1:].reshape(1, -1)

        # Create and train ELM
        elm = ELM(n_input_neurons=n_input, n_hidden_neurons=elm_hidden_neurons, n_output_neurons=n_output)
        elm.train(X_train, y_train)

        elms.append(elm)
        betas.append(elm.get_output_weights()) # Get flattened beta vector

    return elms, betas

def reconstruct_parameter_space(beta_vectors, n_components):
    """
    Applies PCA to the ELM output weight vectors (beta) to find the
    reconstructed parameter space.

    Args:
        beta_vectors: List or array of flattened beta vectors from trained ELMs.
        n_components: Number of principal components to keep (E in paper).

    Returns:
        pca: Fitted PCA object.
        beta_mean: Mean beta vector.
        gamma_vectors: Reconstructed parameter vectors (projections onto PCs).
                       Shape (n_samples, n_components).
    """
    beta_matrix = np.array(beta_vectors) # Shape (n_samples, beta_dim)
    beta_mean = np.mean(beta_matrix, axis=0)
    beta_deviations = beta_matrix - beta_mean

    pca = PCA(n_components=n_components)
    pca.fit(beta_deviations)

    # Calculate gamma: projection of deviations onto principal components
    # gamma = pca.transform(beta_deviations) # This is equivalent if pca is fit on deviations
    # Or using the definition from paper (Eq 9, needs inverse of components)
    # delta_beta = V @ gamma  => gamma = pinv(V) @ delta_beta
    # V are the principal components (eigenvectors), shape (n_components, beta_dim)
    # Need V.T which is (beta_dim, n_components)
    # gamma = beta_deviations @ pinv(pca.components_.T) # Check dimensions carefully
    # Let's use the simpler pca.transform:
    gamma_vectors = pca.transform(beta_deviations) # Shape (n_samples, n_components)

    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {np.sum(pca.explained_variance_ratio_)}")

    return pca, beta_mean, gamma_vectors


def generate_reconstructed_bd(elm_template, pca, beta_mean, gamma_path_points,
                              x0, n_steps_iter, n_transient_iter, n_attractor_points=100):
    """
    Generates the reconstructed bifurcation diagram by iterating the ELM
    along a path in the reconstructed parameter space (gamma).

    Args:
        elm_template: An ELM instance (used for structure and fixed weights).
        pca: Fitted PCA object from reconstruction.
        beta_mean: Mean beta vector.
        gamma_path_points: Points defining the path in gamma space. Shape (n_path_points, n_components).
        x0: Initial condition for iteration.
        n_steps_iter: Total iteration steps per gamma point.
        n_transient_iter: Transient steps to discard per gamma point.
        n_attractor_points: Number of points to keep for plotting BD.

    Returns:
        Dictionary {'params': gamma_values, 'attractor': attractor_points}.
    """
    reconstructed_bd = {'params': [], 'attractor': []}
    n_path_points = gamma_path_points.shape[0]

    for i, gamma in enumerate(gamma_path_points):
        if i % (n_path_points // 10) == 0: # Print progress
             print(f"Generating reconstructed BD point {i+1}/{n_path_points}")

        # Calculate the beta vector for this gamma point
        # delta_beta = gamma @ pca.components_ # If gamma = delta_beta @ V.T
        delta_beta = pca.inverse_transform(gamma.reshape(1, -1)).flatten() # More direct
        current_beta_vector = beta_mean + delta_beta
        current_beta_matrix = current_beta_vector.reshape(elm_template.n_output, elm_template.n_hidden)

        # Iterate the ELM with this beta
        series = elm_template.iterate(x0, n_steps_iter, beta=current_beta_matrix)

        # Keep attractor points
        attractor_pts = series[n_transient_iter:]
        # Limit number of points plotted per param value
        indices_to_keep = np.linspace(0, len(attractor_pts)-1, n_attractor_points, dtype=int)
        attractor_pts_reduced = attractor_pts[indices_to_keep]

        # Store results (use first component of gamma for 1D BD plot)
        param_val_to_plot = gamma[0] if len(gamma) > 0 else i # Use index if 0 components? Should have >=1.
        reconstructed_bd['params'].extend([param_val_to_plot] * len(attractor_pts_reduced))
        reconstructed_bd['attractor'].extend(attractor_pts_reduced)

    return reconstructed_bd