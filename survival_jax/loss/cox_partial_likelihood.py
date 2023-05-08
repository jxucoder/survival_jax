import jax.numpy as jnp
import jax


@jax.jit
def negative_log_cox_partial_likelihood(w, X, indices, riskset, d, lmbd, reg='l1'):
    wx = w @ X.T
    exp_wx = jnp.exp(wx)
    if 'l1':
        reg_term = lmbd * jnp.linalg.norm(w, ord=1) if reg else 0
    return -jnp.sum(indices @ wx - d * jnp.log(riskset @ exp_wx)) + reg_term
