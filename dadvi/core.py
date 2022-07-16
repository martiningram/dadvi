import numpy as np
from typing import NamedTuple, Callable, Optional
from .optimization import optimize_with_hvp
from functools import partial


class DADVIFuns(NamedTuple):

    # Function of eta (variational parameters) and zs (draws)
    # zs should have shape (M, D), where M is number of fixed draws and D is
    # problem dimension.
    kl_est_and_grad_fun: Callable

    # Function of eta, zs, and b, a vector to compute the hvp with
    kl_est_hvp_fun: Optional[Callable]


def find_dadvi_optimum(
    init_params, zs, dadvi_funs, opt_method="trust-ncg", callback_fun=None
):

    val_and_grad_fun = lambda var_params: dadvi_funs.kl_est_and_grad_fun(var_params, zs)
    hvp_fun = (
        None
        if dadvi_funs.kl_est_hvp_fun is None
        else lambda var_params, b: dadvi_funs.kl_est_hvp_fun(var_params, zs, b)
    )

    opt_result, eval_count = optimize_with_hvp(
        val_and_grad_fun,
        hvp_fun,
        init_params,
        opt_method=opt_method,
        callback_fun=callback_fun,
    )

    return {"opt_result": opt_result, "evaluation_count": eval_count}


def get_dadvi_draws(var_params, zs):

    # TODO: Could use JAX here
    means, log_sds = np.split(var_params, 2)
    sds = np.exp(log_sds)

    draws = means.reshape(1, -1) + zs * sds.reshape(1, -1)

    return draws


def compute_lrvb_covariance_direct_method(opt_params, zs, hvp_fun, top_left_only=True):

    rel_hvp_fun = lambda b: hvp_fun(opt_params, zs, b)
    target_vecs = np.eye(opt_params.shape[0])

    # TODO: Check this is correct
    hessian = np.stack([rel_hvp_fun(x) for x in target_vecs])

    # TODO: This could use JAX I guess.
    lrvb_cov_full = np.linalg.inv(hessian)

    if top_left_only:
        n_rel = zs.shape[1]
        return lrvb_cov_full[:n_rel, :n_rel]
    else:
        return lrvb_cov_full
