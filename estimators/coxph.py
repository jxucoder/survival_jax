import numpy as np
from collections import Counter

import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jax import grad
from scipy.optimize import minimize
from survivaljax.estimators.utils import find_biggest_element_less_than_x


class CoxPH:
    """
    Cox Proportial Hazards with jax gradient and hessian
    """
    def __init__(self):

    def get_unique_failure_times(times, events):
        unique_failure_times = set()
        for i, (time, event) in enumerate(zip(times, events)):
            if event:
                unique_failure_times.add(time)
        sorted_times = sorted(unique_failure_times)
        index_dict = {elem: i for i, elem in enumerate(sorted_times)}
        return sorted_times, index_dict

    def create_indices_matrix(times, events):
        # Create dictionary mapping unique times to
        # their indices in the original list
        sorted_times, index_dict = get_unique_failure_times(times, events)
        num_times = len(sorted_times)

        # Get number of items and initialize matrix
        num_samples = len(times)
        matrix = np.zeros((num_times, num_samples), dtype=int)

        # Populate matrix
        for i, (time, event) in enumerate(zip(times, events)):
            if event:
                matrix[index_dict[time]][i] = event
        return jnp.array(matrix)

    def create_risket_matrix(times, events):
        # Create dictionary mapping unique times to their indices in the original list
        sorted_times, index_dict = get_unique_failure_times(times, events)
        num_times = len(sorted_times)

        # Get number of items and initialize matrix
        num_samples = len(times)
        matrix = np.zeros((num_times, num_samples), dtype=int)

        # Populate risk set matrix
        for i, (time, event) in enumerate(zip(times, events)):
            if time in index_dict:
                index = index_dict[time]
            else:
                index = find_biggest_element_less_than_x(sorted_times, time)
            matrix[:index + 1, i] = 1
        return jnp.array(matrix)

    def create_counter_array(times, events):
        # Create dictionary mapping unique times to their indices in the original list
        sorted_times, index_dict = get_unique_failure_times(times, events)
        failure_times = []
        for i, (time, event) in enumerate(zip(times, events)):
            if event:
                failure_times.append(time)
        time_counters = Counter(failure_times)
        counter_array = jnp.array([time_counters[k] for k in sorted_times])

        return counter_array
