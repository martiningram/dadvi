import numpy as np
from .pymc3_to_jax import get_jax_functions_from_pymc3, transform_samples
from dadvi.jax import build_dadvi_funs
from dadvi.core import find_dadvi_optimum, get_dadvi_draws
from jax import vmap


def fit_dadvi_pymc3(pymc3_model, M=50, n_draws=1000):

    jax_funs = get_jax_functions_from_pymc3(pymc3_model)
    dadvi_funs = build_dadvi_funs(jax_funs['log_posterior_fun'])
    init_params = np.zeros(jax_funs['n_params']*2)

    zs = np.random.randn(M, jax_funs['n_params'])
    opt_result = find_dadvi_optimum(init_params, zs, dadvi_funs)

    pred_zs = np.random.randn(n_draws, jax_funs['n_params'])
    pred_draws = get_dadvi_draws(opt_result['opt_result'].x, pred_zs)
    dict_draws = vmap(jax_funs['unflatten_fun'])(pred_draws)

    # Add a "chain" dimension
    dict_draws = {x: np.expand_dims(y, axis=0) for x, y in dict_draws.items()}

    transformed_draws = transform_samples(dict_draws, pymc3_model, keep_untransformed=True)

    return transformed_draws, opt_result