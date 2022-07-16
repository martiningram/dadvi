from jax.flatten_util import ravel_pytree
from theano.link.jax.jax_dispatch import jax_funcify
import theano
import jax
import pymc3 as pm


def get_logp_fn_dict(logp_fn, var_names):
    def logp_fn_dict(theta_dict):

        as_list = [theta_dict[x] for x in var_names]

        return logp_fn(*as_list)

    return logp_fn_dict


def get_basic_init_from_pymc3(pymc3_model):
    # TODO: Improve poor naming here

    from pymc3 import modelcontext

    model = modelcontext(pymc3_model)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = {rv_name: model.test_point[rv_name] for rv_name in rv_names}

    return model, init_state, logp_fn_jax


def get_var_shapes_from_model(pymc3_model):
    # TODO: This probably does too much work.

    _, init_state, _ = get_basic_init_from_pymc3(pymc3_model)

    var_names = list(init_state.keys())
    var_shapes = {x: y.shape for x, y in init_state.items()}
    var_shapes = dict(sorted(var_shapes.items(), key=lambda x: x[0]))

    return var_names, var_shapes


def get_jax_functions_from_pymc3(pymc3_model):

    _, init_state, logp_fn_jax = get_basic_init_from_pymc3(pymc3_model)
    var_names, _ = get_var_shapes_from_model(pymc3_model)
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


def transform_samples(samples, model, keep_untransformed=False):
    # This used to be part of PyMC but is now removed, hence moved to this repo.

    # Find out which RVs we need to compute:
    free_rv_names = {x.name for x in model.free_RVs}
    unobserved_names = {x.name for x in model.unobserved_RVs}

    names_to_compute = unobserved_names - free_rv_names
    ops_to_compute = [x for x in model.unobserved_RVs if x.name in names_to_compute]

    # Create function graph for these:
    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, ops_to_compute)

    # Jaxify, which returns a list of functions, one for each op
    jax_fns = jax_funcify(fgraph)

    # Put together the inputs
    inputs = [samples[x.name] for x in model.free_RVs]

    for cur_op, cur_jax_fn in zip(ops_to_compute, jax_fns):

        # We need a function taking a single argument to run vmap, while the
        # jax_fn takes a list, so:
        result = jax.vmap(jax.vmap(cur_jax_fn))(*inputs)

        # Add to sample dict
        samples[cur_op.name] = result

    # Discard unwanted transformed variables, if desired:
    vars_to_keep = set(
        pm.util.get_default_varnames(
            list(samples.keys()), include_transformed=keep_untransformed
        )
    )
    samples = {x: y for x, y in samples.items() if x in vars_to_keep}

    return samples
