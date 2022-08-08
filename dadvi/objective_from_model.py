import numpy as np
from .core import get_dadvi_draws, DADVIFuns


def build_dadvi_funs(
    log_posterior_fun, log_posterior_grad_fun, log_posterior_hvp_fun=None
):
    """
    Builds DADVIFuns -- i.e., value, gradient and hvp of the DADVI objective --
    from the log posterior density, its gradient and, optionally, its hvp.

    Args:
        log_posterior_fun: The log posterior density. Must take a vector of
            length D and return a scalar, the log posterior density for the
            parameters.
        log_posterior_grad_fun: The gradient of the log posterior density. Must
            take a vector of length D and return a vector of the same length.
        log_posterior_hvp_fun: Optional. If provided, must be a function taking
            two vectors, both of length D, and computing the product of the
            Hessian of the log posterior density, evaluated at the first vector,
            multiplied with the second vector.

    Returns:
    The DADVIFuns as defined in the core module.
    """

    def kl_est_fun(var_params, zs):

        _, log_sds = np.split(var_params, 2)

        draws = get_dadvi_draws(var_params, zs)
        log_probs = np.array([log_posterior_fun(cur_draw) for cur_draw in draws])
        grads = np.array([log_posterior_grad_fun(cur_draw) for cur_draw in draws])

        mean_log_prob = np.mean(log_probs)

        scaled_zs = zs * np.exp(log_sds).reshape(1, -1)
        grad_log_sd = np.mean(scaled_zs * grads, axis=0) + 1
        entropy = np.sum(log_sds)
        grad_of_mean = np.mean(grads, axis=0)

        return -(mean_log_prob + entropy), np.concatenate([-grad_of_mean, -grad_log_sd])

    if log_posterior_hvp_fun is None:
        return DADVIFuns(kl_est_and_grad_fun=kl_est_fun, kl_est_hvp_fun=None)

    # Otherwise, compute hvp
    def kl_est_hvp(var_params, zs, b):

        eta_1, eta_2 = np.split(b, 2)
        mu, log_sigma = np.split(var_params, 2)
        sigma = np.exp(log_sigma)

        full_hvp = np.zeros_like(var_params)

        for cur_z in zs:

            cur_draw = cur_z * sigma + mu
            cur_g = log_posterior_grad_fun(cur_draw)

            sigma_z = sigma * cur_z

            # First part of hvp:
            H_eta_1 = log_posterior_hvp_fun(cur_draw, eta_1)
            H_term_2 = log_posterior_hvp_fun(cur_draw, eta_2 * sigma_z)

            top_part = H_eta_1 + H_term_2

            bottom_term_1 = sigma_z * H_eta_1
            bottom_term_2 = H_term_2 * sigma_z
            bottom_term_3 = cur_g * sigma_z * eta_2

            bottom_part = bottom_term_1 + bottom_term_2 + bottom_term_3

            cur_hvp = np.concatenate([top_part, bottom_part])

            full_hvp += cur_hvp

        return -(full_hvp / zs.shape[0])

    return DADVIFuns(kl_est_fun, kl_est_hvp)
