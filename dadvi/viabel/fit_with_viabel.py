from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from jax import grad, jit
from typing import NamedTuple, Callable
from viabel import bbvi


class StanFitSurrogate(NamedTuple):
    log_prob: Callable
    grad_log_prob: Callable
    constrain_pars: Callable


def fit_pymc_model_with_viabel(
    pymc_model, num_mc_samples=50, n_iters=100000, init_var_param=None
):

    jax_funs = get_jax_functions_from_pymc(pymc_model)

    post_fun = jit(jax_funs["log_posterior_fun"])
    grad_fun = jit(grad(jax_funs["log_posterior_fun"]))

    for_viabel = StanFitSurrogate(
        log_prob=post_fun, grad_log_prob=grad_fun, constrain_pars=lambda x: x
    )

    mf_results = bbvi(
        jax_funs["n_params"],
        fit=for_viabel,
        num_mc_samples=num_mc_samples,
        n_iters=n_iters,
        init_var_param=init_var_param,
    )

    return mf_results
