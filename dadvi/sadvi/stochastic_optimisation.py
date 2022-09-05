from typing import NamedTuple, Callable


class StochasticOptimizer(NamedTuple):

    # Called with n_params, **kwargs
    initialise_state: Callable

    # Called with params, grad, state, **kwargs
    update_params_and_state: Callable
