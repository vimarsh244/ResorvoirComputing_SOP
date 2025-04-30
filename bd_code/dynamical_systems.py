import numpy as np

def logistic_map(p, x):
  """Applies the logistic map formula."""
  return p * x * (1.0 - x)

def sine_map(p1, p2, x, c1=0.8, c2=1.25):
  """Applies the sine map formula (as used in the paper)."""
  # Paper uses c1=0.8, c2=1.25, p2=0 for 1D reconstruction
  return c1 * np.sin(c2 * np.pi * p1 * x) + p2

def generate_time_series(map_func, params, x0, n_steps, n_transient, noise_level=0.0, noise_type='observational'):
  """
  Generates time series data from a given map.

  Args:
      map_func: The map function (e.g., logistic_map, sine_map).
      params: A dictionary or list/tuple of parameters for the map_func.
      x0: Initial condition.
      n_steps: Total number of steps to generate (after transient).
      n_transient: Number of initial steps to discard.
      noise_level: Standard deviation of Gaussian noise to add.
      noise_type: 'observational' (added after generation) or 'dynamical' (added each step).

  Returns:
      Time series array (n_steps,).
  """
  x = np.zeros(n_steps + n_transient)
  x[0] = x0
  current_params = params if isinstance(params, (list, tuple)) else params.values()

  for t in range(n_steps + n_transient - 1):
      xt = x[t]
      if noise_type == 'dynamical' and noise_level > 0:
         # Apply dynamical noise before map (or sometimes after, depending on model)
         # Let's add it after the map calculation for simplicity here, matching paper's eq for logistic noise
         # x[t+1] = map_func(*current_params, xt) + np.random.normal(0, noise_level)
         # Paper text for logistic seems to add noise *after* map calculation: x(t+1) = p*x(t)*(1-x(t)) + noise
          x[t+1] = map_func(*current_params, xt) + np.random.normal(0, noise_level)
          # Clip to prevent divergence for logistic map
          if map_func == logistic_map:
              x[t+1] = np.clip(x[t+1], 0, 1)

      else:
          x[t+1] = map_func(*current_params, xt)

  # Discard transient
  series = x[n_transient:]

  if noise_type == 'observational' and noise_level > 0:
      series += np.random.normal(0, noise_level, size=n_steps)
      # Clip again if needed
      if map_func == logistic_map:
            series = np.clip(series, 0, 1)

  # Normalize data (as mentioned in paper for circuit data)
  # Although paper applies this *after* measurement, applying it here
  # ensures ELM works on a consistent scale.
  # min_val = np.min(series)
  # max_val = np.max(series)
  # if max_val > min_val:
  #     series = (series - min_val) / (max_val - min_val)
  # Let's skip normalization for now to directly match map outputs,
  # but be aware it might be needed for real/noisy data.

  return series

def generate_bifurcation_data(map_func, p_values, param_index, other_params, x0, n_steps, n_transient, noise_level=0.0, noise_type='observational', n_attractor_points=100):
    """Generates data points for a bifurcation diagram."""
    bifurcation_dict = {'params': [], 'attractor': []}
    if not isinstance(p_values, (np.ndarray, list)):
        p_values = [p_values] # Handle single parameter case

    for p in p_values:
        current_params_list = list(other_params) # Make a mutable list
        current_params_list.insert(param_index, p)

        series = generate_time_series(map_func, current_params_list, x0, n_steps, n_transient, noise_level, noise_type)

        # Keep only the last points assumed to be on the attractor
        attractor_points = series[-n_attractor_points:]

        bifurcation_dict['params'].extend([p] * n_attractor_points)
        bifurcation_dict['attractor'].extend(attractor_points)

    return bifurcation_dict