from typing import Callable, Dict, Tuple
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, jvp, grad
from functools import partial
from dadvi.core import DADVIFuns, compute_preconditioner_from_var_params
from dadvi.utils import cg_using_fun_scipy


@partial(jit, static_argnums=0)
def hvp(f, primals, tangents):
    # Taken (and slightly modified) from:
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    return jvp(grad(f), (primals,), (tangents,))[1]


@jit
def _make_draws(z, mean, log_sd):

    draw = z * jnp.exp(log_sd) + mean

    return draw


@jit
def _calculate_entropy(log_sds):

    return jnp.sum(log_sds)


def build_dadvi_funs(log_posterior_fn: Callable[[jnp.ndarray], float]) -> DADVIFuns:
    """
    Builds the DADVIFuns from a log posterior density function written in JAX.
    """

    def single_log_posterior_fun(cur_z, var_params):
        means, log_sds = jnp.split(var_params, 2)
        cur_theta = _make_draws(cur_z, means, log_sds)
        return log_posterior_fn(cur_theta)

    def log_posterior_expectation(zs, var_params):
        single_curried = partial(single_log_posterior_fun, var_params=var_params)
        log_posts = vmap(single_curried)(zs)
        return jnp.mean(log_posts)

    def full_kl_est(var_params, zs):
        _, log_sds = jnp.split(var_params, 2)
        log_posterior = log_posterior_expectation(zs, var_params)
        entropy = _calculate_entropy(log_sds)
        return -log_posterior - entropy

    @jit
    def kl_est_hvp_fun(var_params, zs, b):
        rel_kl_est = partial(full_kl_est, zs=zs)
        rel_hvp = lambda x, y: hvp(rel_kl_est, x, y)
        return rel_hvp(var_params, b)

    kl_est_and_grad_fun = jit(value_and_grad(full_kl_est))

    return DADVIFuns(
        kl_est_and_grad_fun=kl_est_and_grad_fun, kl_est_hvp_fun=kl_est_hvp_fun
    )


def compute_posterior_mean_and_sd_using_cg_delta_method(
    fun_to_evaluate: Callable[[Dict[str, jnp.ndarray]], float],
    final_var_params: jnp.ndarray,
    fixed_draws: jnp.ndarray,
    dadvi_funs: DADVIFuns,
    unflatten_fun: Callable[[jnp.ndarray], Dict[str, jnp.ndarray]],
) -> Tuple[float, float]:
    """
    Params:
        fun_to_evaluate: The function of interest. Takes a dictionary of parameter values
            and should return a scalar.
        final_var_params: The variational parameters to use, as a flat vector whose first half
            contains the means.
        fixed_draws: The fixed draws to use, as a matrix of shape [M, n_params].
        dadvi_funs: The DADVIFuns to use.
        unflatten_fun: The function mapping from the flat parameters to a dictionary of
            param_names -> values.
    """

    def fun_to_differentiate(x):

        means = jnp.split(x, 2)[0]
        unflattened = unflatten_fun(means)
        result = fun_to_evaluate(unflattened)

        return result

    rel_mean, rel_grad = value_and_grad(fun_to_differentiate)(final_var_params)
    preconditioner = compute_preconditioner_from_var_params(final_var_params)

    rel_hvp = lambda b: dadvi_funs.kl_est_hvp_fun(final_var_params, fixed_draws, b)

    h_inv_g, succ = cg_using_fun_scipy(rel_hvp, rel_grad, preconditioner)

    assert succ == 0

    var_est = rel_grad @ h_inv_g
    sd_est = jnp.sqrt(var_est)

    return rel_mean, sd_est
