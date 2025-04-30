import numpy as np
from scipy.spatial import KDTree
from dynamical_systems import logistic_map, sine_map # For derivative method
from elm import ELM # For derivative method

def lyapunov_exponent_ts(series, embedding_dim=1, time_delay=1, mean_period=1, min_neighbors=5, epsilon=None, max_steps=None):
    """
    Estimate the largest Lyapunov exponent from a time series using a
    method similar to Rosenstein et al. / Kantz & Schreiber / Yao et al.

    Args:
        series (np.ndarray): The input time series.
        embedding_dim (int): Embedding dimension (1 for 1D maps).
        time_delay (int): Time delay for embedding (1 for maps).
        mean_period (int): Estimated mean period to avoid neighbors from the same cycle.
        min_neighbors (int): Minimum number of neighbors to consider.
        epsilon (float, optional): Maximum distance for neighbors. If None, uses a fraction of data std dev.
        max_steps (int, optional): Number of steps to track divergence. If None, uses a fraction of series length.

    Returns:
        float: Estimated largest Lyapunov exponent.
        np.ndarray: Time steps (for plotting divergence curve).
        np.ndarray: Mean log divergence at each time step.
    """
    if embedding_dim != 1 or time_delay != 1:
        print("Warning: This LE implementation is simplified for 1D maps (embedding_dim=1, time_delay=1).")

    N = len(series)
    if max_steps is None:
        max_steps = N // 10 # Heuristic

    # Simplified embedding for dim=1
    trajectory = series.reshape(-1, 1)

    # Use KDTree for efficient neighbor search
    tree = KDTree(trajectory)

    if epsilon is None:
         # Heuristic for epsilon based on data range or std dev
         epsilon = (np.max(series) - np.min(series)) * 0.025 # Paper uses 0.025, assume normalized? Let's use fraction of std dev
         # epsilon = np.std(series) * 0.1
         if epsilon == 0: epsilon = 0.01 # Avoid zero epsilon for constant series

    log_divergence = []
    valid_indices = []

    # Iterate through points in the trajectory as reference points
    # Avoid edges where we can't track divergence for max_steps
    for i in range(N - max_steps):
        ref_point = trajectory[i:i+1] # Keep 2D shape for KDTree query
        # Find neighbors within epsilon, excluding self and temporally close points
        # query_ball_point returns list of indices
        neighbor_indices = tree.query_ball_point(ref_point, r=epsilon, p=2)[0] # p=2 for Euclidean

        # Filter out self and temporally close neighbors
        valid_neighbors = []
        for idx in neighbor_indices:
            # Exclude self: idx != i
            # Exclude temporally close: abs(idx - i) > mean_period
            # Exclude points too close to the end: idx < N - max_steps
            if idx != i and abs(idx - i) > mean_period and idx < N - max_steps:
                valid_neighbors.append(idx)

        if len(valid_neighbors) >= min_neighbors:
            distances_at_step = []
            # Track divergence for max_steps
            for k in range(max_steps):
                initial_dist_sum = 0
                evolved_dist_sum = 0
                pair_count = 0
                # Calculate average distance evolution for *pairs* (as in paper description)
                # This differs slightly from Rosenstein (avg log dist) but matches paper's d(t) description better
                # We use pairs of valid neighbors here, rather than pairs involving the reference point i
                # Let's try the Rosenstein/Kantz way: distance from reference
                current_step_log_dist = []
                for neighbor_idx in valid_neighbors:
                     dist = np.abs(trajectory[i+k, 0] - trajectory[neighbor_idx+k, 0])
                     if dist > 1e-12: # Avoid log(0)
                         current_step_log_dist.append(np.log(dist))

                if current_step_log_dist:
                     distances_at_step.append(np.mean(current_step_log_dist))
                else:
                     # If no valid distances, stop tracking for this ref point
                     distances_at_step.append(np.nan) # Use NaN to indicate break
                     break

            # Only add if we got a full divergence curve
            if len(distances_at_step) == max_steps and not np.isnan(distances_at_step).any():
                log_divergence.append(distances_at_step)
                valid_indices.append(i)


    if not log_divergence:
        print("Warning: Could not find enough valid neighbor pairs to estimate LE.")
        return 0.0, np.arange(max_steps), np.full(max_steps, np.nan) # Return 0 or NaN? Paper implies 0 for periodic

    log_divergence_arr = np.array(log_divergence) # Shape (n_valid_indices, max_steps)
    mean_log_divergence = np.mean(log_divergence_arr, axis=0)

    # Estimate LE from the slope of the linear region
    steps = np.arange(max_steps)
    # Find a reasonable linear region (heuristic: first 10-20% of steps?)
    fit_end = max(5, max_steps // 10)
    if fit_end <= 1: fit_end = max_steps # Handle very short series

    coeffs = np.polyfit(steps[1:fit_end], mean_log_divergence[1:fit_end], 1)
    le_estimate = coeffs[0]

    # Paper sets LE=0 for periodic orbits where d(t)=0.
    # Check if the series looks periodic (e.g., very small standard deviation after transient)
    if np.std(series[-len(series)//2:]) < 1e-6: # Heuristic check
        # print("Series appears periodic, setting LE to 0.")
        return 0.0, steps, mean_log_divergence


    # Note: Paper formula (Eq 12) uses sum(ln(d(t+1)/d(t))), which is related but different.
    # The slope of mean(ln(d(t))) vs t is a more standard approach (Kantz & Schreiber).
    # Let's stick to the slope method.

    return le_estimate, steps, mean_log_divergence


def lyapunov_exponent_derivative(map_func, params, x0, n_steps, n_transient):
    """
    Estimate the Lyapunov exponent using the map's derivative.

    Args:
        map_func: The map function (e.g., logistic_map).
        params: Parameters for the map function.
        x0: Initial condition.
        n_steps: Number of steps for averaging.
        n_transient: Number of transient steps.

    Returns:
        float: Estimated Lyapunov exponent.
    """
    x = x0
    current_params = params if isinstance(params, (list, tuple)) else params.values()

    # Run through transient
    for _ in range(n_transient):
        x = map_func(*current_params, x)

    log_deriv_sum = 0.0
    for _ in range(n_steps):
        # Calculate derivative at current x
        if map_func == logistic_map:
            p = current_params[0]
            deriv = p * (1.0 - 2.0 * x)
        elif map_func == sine_map:
            p1, p2 = current_params[0], current_params[1]
            c1, c2 = 0.8, 1.25 # Match paper's constants
            deriv = c1 * np.cos(c2 * np.pi * p1 * x) * (c2 * np.pi * p1)
        else:
            raise NotImplementedError("Derivative not implemented for this map.")

        # Avoid issues with derivative being zero or negative
        if abs(deriv) > 1e-12:
             log_deriv_sum += np.log(np.abs(deriv))
        else:
             # If derivative is ~0, contribution to LE is -inf, orbit is superstable
             # Return a large negative number or handle as a special case
             return -np.inf # Or handle based on context

        # Iterate map
        x = map_func(*current_params, x)

    return log_deriv_sum / n_steps

def lyapunov_exponent_elm_derivative(elm_instance, beta, x0, n_steps, n_transient):
    """
    Estimate LE using the ELM's effective derivative.

    Args:
        elm_instance: The trained ELM instance (needed for W_in, biases, sigmoid').
        beta: The output weights (beta) defining the specific ELM map. Shape (n_output, n_hidden).
        x0: Initial condition.
        n_steps: Number of steps for averaging.
        n_transient: Number of transient steps.

    Returns:
        float: Estimated Lyapunov exponent for the ELM map.
    """
    x = x0
    log_deriv_sum = 0.0
    current_x_arr = np.array([[x]]) # Shape (1, 1)

    # Sigmoid derivative: s'(z) = s(z) * (1 - s(z))
    def sigmoid_derivative(z):
         s_z = elm_instance._sigmoid(z)
         return s_z * (1.0 - s_z)

    # Run through transient using ELM iteration
    for _ in range(n_transient):
        H_current = elm_instance._hidden_layer_output(current_x_arr)
        current_x_arr = beta @ H_current # Shape (n_output, 1)

    # Average log derivative along trajectory
    for _ in range(n_steps):
        # Calculate derivative dg/dx = beta @ diag(s'(Wx+b)) @ W
        # x is current_x_arr, shape (1, 1)
        linear_output = elm_instance.input_weights @ current_x_arr + elm_instance.biases # Shape (n_hidden, 1)
        s_prime_vals = sigmoid_derivative(linear_output) # Shape (n_hidden, 1)

        # Need element-wise product then matrix multiply: beta @ (s_prime * W)
        # W shape (n_hidden, n_input) -> (n_hidden, 1) for n_input=1
        # beta shape (n_output, n_hidden) -> (1, n_hidden) for n_output=1
        # s_prime shape (n_hidden, 1)
        # Result needs to be scalar for 1D map

        # Correct calculation: Sum over hidden neurons k: sum_k [ beta_k * s'(Wx+b)_k * W_k ]
        term_inside_sum = beta * s_prime_vals.T * elm_instance.input_weights.T # Element-wise products broadcast correctly?
        # beta (1, n_hidden)
        # s_prime.T (1, n_hidden)
        # W.T (1, n_hidden) for n_input=1
        deriv_val = np.sum(term_inside_sum) # Sum across hidden dimension

        if abs(deriv_val) > 1e-12:
             log_deriv_sum += np.log(np.abs(deriv_val))
        else:
             return -np.inf # Superstable point

        # Iterate ELM
        H_current = elm_instance._hidden_layer_output(current_x_arr)
        current_x_arr = beta @ H_current

    return log_deriv_sum / n_steps