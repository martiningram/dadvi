import numpy as np
from .core import get_dadvi_draws


def get_objective_fun(log_posterior_fun, log_posterior_grad_fun):

    def kl_est_fun(var_params, zs):

        _, log_sds = np.split(var_params, 2)

        draws = get_dadvi_draws(var_params, zs)
        log_probs = np.array([log_posterior_fun(cur_draw) for cur_draw in draws])
        grads = np.array([log_posterior_grad_fun(cur_draw) for cur_draw in draws])

        mean_log_prob = np.mean(log_probs)

        scaled_zs = zs * np.exp(log_sds).reshape(-1, 1)
        grad_log_sd = np.mean(scaled_zs * grads, axis=0) + 1
        entropy = np.sum(log_sds)
        grad_of_mean = np.mean(grads, axis=0)

        return -(mean_log_prob + entropy), np.concatenate([-grad_of_mean, -grad_log_sd])

    return kl_est_fun
