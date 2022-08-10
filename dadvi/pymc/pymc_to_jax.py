from jax.flatten_util import ravel_pytree
from pymc.sampling_jax import get_jaxified_logp


def get_logp_fn_dict(logp_fn, var_names):
    def logp_fn_dict(theta_dict):

        as_list = [theta_dict[x] for x in var_names]

        return logp_fn(as_list)

    return logp_fn_dict


def get_basic_init_from_pymc(pymc_model):

    logp_fn_jax = get_jaxified_logp(pymc_model)

    rv_names = [rv.name for rv in pymc_model.value_vars]
    init_point = pymc_model.initial_point()
    init_state = {rv_name: init_point[rv_name] for rv_name in rv_names}

    return rv_names, init_state, logp_fn_jax


def get_jax_functions_from_pymc(pymc_model):

    var_names, init_state, logp_fn_jax = get_basic_init_from_pymc(pymc_model)
    logp_fn_dict = get_logp_fn_dict(logp_fn_jax, var_names)
    flat_init, fun = ravel_pytree(init_state)

    def flat_log_post_fun(flat_params):

        param_dict = fun(flat_params)
        return logp_fn_dict(param_dict)

    return {
        "log_posterior_fun": flat_log_post_fun,
        "unflatten_fun": fun,
        "n_params": flat_init.shape[0],
    }
