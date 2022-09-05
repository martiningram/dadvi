import jax.numpy as jnp
from jax import jit
from .stochastic_optimisation import StochasticOptimizer


def initialise_state(n_params, n_win=10, i=0):

    windowed_grads = jnp.zeros((n_params, n_win))

    return {"i": i, "windowed_grads": windowed_grads, "n_win": n_win}


@jit
def update_params_and_state(
    cur_params, cur_grad, opt_state, learning_rate=0.001, epsilon=0.1
):

    new_entry_num = opt_state["i"] % opt_state["n_win"]

    windowed_grads = opt_state["windowed_grads"]

    transposed_grads = windowed_grads.T
    transposed_grads = transposed_grads.at[new_entry_num].set(cur_grad**2)
    windowed_grads = transposed_grads.T

    # windowed_grads[:, new_entry_num] = cur_grad ** 2
    # windowed_grads = index_update(windowed_grads.T, new_entry_num, cur_grad**2).T

    summed_grads = windowed_grads.sum(axis=1)

    update = learning_rate * cur_grad / jnp.sqrt(summed_grads + epsilon)

    new_params = cur_params - update
    new_i = opt_state["i"] + 1

    new_state = {
        "i": new_i,
        "windowed_grads": windowed_grads,
        "n_win": opt_state["n_win"],
    }

    return new_params, new_state


windowed_adagrad = StochasticOptimizer(
    initialise_state=initialise_state, update_params_and_state=update_params_and_state
)
