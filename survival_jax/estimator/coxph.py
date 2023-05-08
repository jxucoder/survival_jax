import numpy as np

import jax.numpy as jnp
from jax import grad, hessian
from scipy.optimize import minimize
from survival_jax.loss.cox_partial_likelihood import negative_log_cox_partial_likelihood
from functools import partial
from survival_jax.estimator.utils import (
    create_indices_matrix,
    create_risket_matrix,
    create_counter_array,
)


class CoxPH:
    """
    Cox Proportional Hazards with jax gradient and hessian
    """
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray, lmbd: float = 0.1):
        indices = create_indices_matrix(times, events)
        riskset = create_risket_matrix(times, events)
        d = create_counter_array(times, events)
        covariates = jnp.array(X)

        cost_func = partial(negative_log_cox_partial_likelihood, indices=indices, riskset=riskset, d=d, X=covariates,
                            lmbd=lmbd)
        g = grad(cost_func)
        h = hessian(cost_func)

        result = minimize(cost_func, np.zeros(14), method='Newton-CG', jac=g, hess=h)
        betas = result.x
        return betas
