import numpy as np
from collections import Counter

import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jax import grad, hessian
from scipy.optimize import minimize
from survival_jax.estimator.utils import find_biggest_element_less_than_x
from survival_jax.loss.cox_partial_likelihood import negative_log_cox_partial_likelihood
from functools import partial


class CoxPH:
    """
    Cox Proportional Hazards with jax gradient and hessian
    """
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray, lmbd: float = 0.1):
        indices = self.create_indices_matrix(times, events)
        riskset = self.create_risket_matrix(times, events)
        d = self.create_counter_array(times, events)
        covariates = jnp.array(X)

        cost_func = partial(negative_log_cox_partial_likelihood, indices=indices, riskset=riskset, d=d, X=covariates,
                            lmbd=lmbd)
        g = grad(cost_func)
        h = hessian(cost_func)

        result = minimize(cost_func, np.zeros(14), method='Newton-CG', jac=g, hess=h)
        betas = result.x
        return betas

    def get_unique_failure_times(self, times, events):
        unique_failure_times = set()
        for i, (time, event) in enumerate(zip(times, events)):
            if event:
                unique_failure_times.add(time)
        sorted_times = sorted(unique_failure_times)
        index_dict = {elem: i for i, elem in enumerate(sorted_times)}
        return sorted_times, index_dict

    def create_indices_matrix(self, times, events):
        # Create dictionary mapping unique times
        # row i, col j: if j-th sample fails at i-th unique failure time, t_i
        # we can look up t_i using index_dict from get_unique_failure_times()
        sorted_times, index_dict = self.get_unique_failure_times(times, events)
        num_times = len(sorted_times)

        # Get number of samples and initialize matrix
        num_samples = len(times)
        matrix = np.zeros((num_times, num_samples), dtype=int)

        for i, (time, event) in enumerate(zip(times, events)):
            if event:
                matrix[index_dict[time]][i] = event
        return jnp.array(matrix)

    def create_risket_matrix(self, times, events):
        # row i, col j: if j-th sample is at risk at i-th unique failure time, t_i
        # we can look up t_i using index_dict from get_unique_failure_times()
        sorted_times, index_dict = self.get_unique_failure_times(times, events)
        num_times = len(sorted_times)

        # Get number of samples and initialize matrix
        num_samples = len(times)
        matrix = np.zeros((num_times, num_samples), dtype=int)

        # Populate riskset matrix
        for i, (time, event) in enumerate(zip(times, events)):
            if time in index_dict:
                index = index_dict[time]
            else:
                index = find_biggest_element_less_than_x(sorted_times, time)
            matrix[:index + 1, i] = 1
        return jnp.array(matrix)

    def create_counter_array(self, times, events):
        # i-th item: numbers of samples failed at t_i
        sorted_times, index_dict = self.get_unique_failure_times(times, events)
        failure_times = []
        for i, (time, event) in enumerate(zip(times, events)):
            if event:
                failure_times.append(time)
        time_counters = Counter(failure_times)
        counter_array = jnp.array([time_counters[k] for k in sorted_times])
        return counter_array
