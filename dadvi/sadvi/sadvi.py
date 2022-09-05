import jax.numpy as jnp
import numpy as np
import time
from jax.nn import softplus
from jax import jit, value_and_grad, vmap
from functools import partial
from .windowed_adagrad import windowed_adagrad
from tqdm import tqdm


def compute_objective(var_params, z, log_p_fn, use_softplus=False):

    if use_softplus:
        var_means, rhos = jnp.split(var_params, 2)
        sds = softplus(rhos)
        cur_entropy = jnp.sum(jnp.log(sds))
        cur_draw = sds * z + var_means
    else:
        var_means, var_log_sds = jnp.split(var_params, 2)
        cur_entropy = jnp.sum(var_log_sds)
        cur_draw = jnp.exp(var_log_sds) * z + var_means

    cur_log_p = log_p_fn(cur_draw)

    return -cur_log_p - cur_entropy


def kl_estimate_callback(
    i,
    cur_params,
    cur_state,
    mean_elbo,
    mean_grad,
    averaged_objective_fun,
    compute_every=1000,
    n_draws=1000,
    verbose=True,
):

    if i == 0:

        kl_estimate_callback.estimates = list()

    if i % compute_every == 0:

        n_model_params = cur_params.shape[0] // 2
        zs = np.random.randn(n_draws, n_model_params)
        mean_est = averaged_objective_fun(zs, cur_params)

        if verbose:
            print(i, mean_est)

        kl_estimate_callback.estimates.append({"i": i, "kl_est": mean_est})


def parameter_history_callback(
    i,
    cur_params,
    cur_state,
    mean_elbo,
    mean_grad,
    averaged_objective_fun,
    compute_every=100,
):

    if i == 0:
        parameter_history_callback.param_history = list()

    if i % compute_every == 0:
        parameter_history_callback.param_history.append(
            {"i": i, "time": time.time(), "params": cur_params}
        )


def relative(current, prev, eps=1e-6):
    # From https://github.com/pymc-devs/pymc/blob/main/pymc/variational/callbacks.py
    return (np.abs(current - prev) + eps) / (np.abs(prev) + eps)


def absolute(current, prev):
    # From https://github.com/pymc-devs/pymc/blob/main/pymc/variational/callbacks.py
    return np.abs(current - prev)


def parameter_change_callback(
    i,
    cur_params,
    cur_state,
    mean_elbo,
    mean_grad,
    averaged_objective_fun,
    every=100,
    diff="relative",
    ord=np.inf,
    tolerance=1e-3,
):

    # Implementation of PyMC's convergence criterion: https://github.com/pymc-devs/pymc/blob/main/pymc/variational/callbacks.py#L38

    if i == 0:
        # Initialise parameters
        parameter_change_callback.prev = cur_params
        parameter_change_callback.norm_hist = list()
        return

    if i % every != 0:
        return

    current, prev = cur_params, parameter_change_callback.prev

    rel_delta = relative(current, prev)
    abs_delta = absolute(current, prev)

    parameter_change_callback.prev = current

    rel_norm = np.linalg.norm(rel_delta, ord)
    abs_norm = np.linalg.norm(abs_delta, ord)

    parameter_change_callback.norm_hist.append(
        {"i": i, "rel_norm": rel_norm, "abs_norm": abs_norm}
    )

    assert diff in ["relative", "absolute"]

    norm_to_use = rel_norm if diff == "relative" else abs_norm
    converged = norm_to_use < tolerance

    return converged


def timing_callback(
    i,
    cur_params,
    cur_state,
    mean_elbo,
    mean_grad,
    averaged_objective_fun,
    compute_every=1,
):

    if i == 0:
        timing_callback.history = list()

    if i % compute_every == 0:
        timing_callback.history.append({"i": i, "time": time.time()})

    return


def relative_kl_callback(
    i,
    cur_params,
    cur_state,
    mean_elbo,
    mean_grad,
    averaged_objective_fun,
    compute_every=100,
    n_draws=100,
):

    if i == 0:

        relative_kl_callback.prev_elbo = None
        relative_kl_callback.history = list()

    if i % compute_every != 0:
        return

    n_model_params = cur_params.shape[0] // 2
    zs = np.random.randn(n_draws, n_model_params)

    cur_elbo = averaged_objective_fun(zs, cur_params)
    prev_elbo = relative_kl_callback.prev_elbo

    if prev_elbo is None:
        # We're on the first iteration
        cur_delta = np.nan
    else:
        cur_delta = np.abs((cur_elbo - prev_elbo) / prev_elbo)

    relative_kl_callback.prev_elbo = cur_elbo

    relative_kl_callback.history.append({"i": i, "rel_delta_elbo": cur_delta})


def initialise_params_and_functions(
    log_p_fn, use_softplus, n_model_params, init_var_params=None
):

    cur_objective = jit(
        partial(compute_objective, log_p_fn=log_p_fn, use_softplus=use_softplus)
    )

    val_and_grad_fn = jit(value_and_grad(cur_objective))

    # TODO: Maybe want to make clear that these sds are either log sds or
    # pre-softplus sds
    if init_var_params is not None:
        var_means, var_sds = np.split(init_var_params, 2)
    else:
        var_means = jnp.zeros(n_model_params)
        var_sds = jnp.repeat(-3, n_model_params)

    var_params = jnp.concatenate([var_means, var_sds])

    @jit
    def averaged_objective(zs, params):

        vals = vmap(lambda z: cur_objective(params, z))(zs)
        return jnp.mean(vals)

    @jit
    def averaged_grad(zs, params):

        vals, grads = vmap(lambda z: val_and_grad_fn(params, z))(zs)
        mean_grad = grads.mean(axis=0)
        mean_elbo = vals.mean(axis=0)

        return mean_elbo, mean_grad

    return {
        "var_params": var_params,
        "averaged_grad": averaged_grad,
        "averaged_objective": averaged_objective,
    }


def run_step(
    i,
    draws_per_step,
    cur_params,
    cur_state,
    optimizer,
    averaged_grad,
    averaged_objective,
    callback_funs=[],
):

    # TODO: Maybe pass in
    n_model_params = cur_params.shape[0] // 2

    z = np.random.randn(draws_per_step, n_model_params)

    mean_elbo, mean_grad = averaged_grad(z, cur_params)

    cur_params, cur_state = optimizer.update_params_and_state(
        cur_params, mean_grad, cur_state
    )

    for callback_fun in callback_funs:
        # TODO: Support early stopping
        callback_fun(i, cur_params, cur_state, mean_elbo, mean_grad, averaged_objective)

    return cur_params, cur_state


def fit_s_advi(
    log_p_fn,
    n_model_params,
    n_steps=100000,
    draws_per_step=1,
    use_softplus=False,
    seed=2,
    callback_funs=[],
    init_var_params=None,
    optimizer=windowed_adagrad,
    opt_init_kwargs={},
    show_progress=False,
):

    # TODO: Allow passing in the DADVI funs rather than requiring a JAX function.
    np.random.seed(seed)

    init = initialise_params_and_functions(
        log_p_fn,
        use_softplus=use_softplus,
        n_model_params=n_model_params,
        init_var_params=init_var_params,
    )

    init_state = optimizer.initialise_state(
        init["var_params"].shape[0], **opt_init_kwargs
    )
    cur_params, cur_state = init["var_params"], init_state

    iterator = tqdm(range(n_steps)) if show_progress else range(n_steps)

    for i in iterator:

        cur_params, cur_state = run_step(
            i,
            draws_per_step,
            cur_params,
            cur_state,
            optimizer,
            init["averaged_grad"],
            init["averaged_objective"],
            callback_funs=callback_funs,
        )

    final_means, final_sds = jnp.split(cur_params, 2)
    final_sds = softplus(final_sds) if use_softplus else jnp.exp(final_sds)

    return {"means": final_means, "sds": final_sds}
