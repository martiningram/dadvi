from jax.flatten_util import ravel_pytree
from pymc.sampling_jax import get_jaxified_logp, get_jaxified_graph
import jax
from pymc.util import get_default_varnames
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
    """
    Given a PyMC model, builds functions for computing posterior densities with JAX.
    Args:
        pymc_model: The PyMC model object.
    Returns:
    A dictionary containing three elements: "log_posterior_fun" is the log posterior
    density, as a function of a flat parameter vector; "unflatten_fun" turns a flat
    parameter vector back into a dictionary; and "n_params" is the number of parameters
    in the model.
    """

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


def transform_dadvi_draws(
    pymc_model,
    flat_dadvi_draws,
    unflatten_fun,
    keep_untransformed=False,
    add_chain_dim=False,
):
    # TODO: Maybe should take unflattened draws as input instead

    non_flat = jax.vmap(unflatten_fun)(flat_dadvi_draws)
    list_version = [non_flat[x.name] for x in pymc_model.value_vars]

    var_names = pymc_model.unobserved_value_vars

    vars_to_sample = list(
        get_default_varnames(var_names, include_transformed=keep_untransformed)
    )

    jax_fn = get_jaxified_graph(inputs=pymc_model.value_vars, outputs=vars_to_sample)

    list_res = jax.vmap(jax_fn)(*list_version)
    samples = {v.name: r for v, r in zip(vars_to_sample, list_res)}

    if add_chain_dim:
        samples = {x: np.expand_dims(y, axis=0) for x, y in samples.items()}

    return samples


def get_flattened_indices_and_param_names(jax_funs):
    init_means = np.zeros(jax_funs["n_params"])

    unflattened_params = jax_funs["unflatten_fun"](init_means)
    params_flattened_individually = {
        x: y.flatten() for x, y in unflattened_params.items()
    }
    encoder = LabelEncoder()
    encoder.fit(list(params_flattened_individually.keys()))
    param_indices = {
        x: np.arange(y.shape[0]) for x, y in params_flattened_individually.items()
    }
    param_names = {
        x: encoder.transform(np.repeat(x, y.shape[0])) for x, y in param_indices.items()
    }
    params_reshaped_again = {
        x: y.reshape(unflattened_params[x].shape) for x, y in param_indices.items()
    }
    indices_reshaped_again = {
        x: y.reshape(unflattened_params[x].shape) for x, y in param_names.items()
    }
    flat_param_indices, _ = ravel_pytree(params_reshaped_again)
    flat_param_names_encoded, unflatten_b = ravel_pytree(indices_reshaped_again)
    flat_param_names = encoder.inverse_transform(flat_param_names_encoded)

    # Make sure these agree
    test_a = unflatten_b(flat_param_names_encoded)
    test_b = jax_funs["unflatten_fun"](flat_param_names_encoded)
    assert all([np.allclose(test_a[x], test_b[x]) for x in test_a])

    return flat_param_indices, flat_param_names
