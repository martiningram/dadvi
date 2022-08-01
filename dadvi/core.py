import numpy as np
from typing import NamedTuple, Callable, Optional
from .optimization import optimize_with_hvp
from functools import partial
from .utils import cg_using_fun_scipy
from scipy.sparse import diags


class DADVIFuns(NamedTuple):
    """
    This NamedTuple holds the functions required to run DADVI.

    Args:
    kl_est_and_grad_fun: Function of eta (variational parameters) and zs (draws).
        zs should have shape (M, D), where M is number of fixed draws and D is
        problem dimension. Returns a tuple whose first argument is the estimate
        of the KL divergence, and the second is its gradient w.r.t. eta.
    kl_est_hvp_fun: Function of eta, zs, and b, a vector to compute the hvp
        with. This should return a vector -- the result of the hvp with b.
    """

    kl_est_and_grad_fun: Callable
    kl_est_hvp_fun: Optional[Callable]


def find_dadvi_optimum(
        init_params, zs, dadvi_funs, opt_method="trust-ncg", callback_fun=None, verbose=False
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
        verbose=verbose
    )

    return {"opt_result": opt_result, "evaluation_count": eval_count}


def get_dadvi_draws(var_params, zs):
    """
    Computes draws from the variational approximation given variational
    parameters and a matrix of fixed draws.
    """

    # TODO: Could use JAX here
    means, log_sds = np.split(var_params, 2)
    sds = np.exp(log_sds)

    draws = means.reshape(1, -1) + zs * sds.reshape(1, -1)

    return draws


def get_lrvb_draws(opt_means, lrvb_cov, zs):

    # Check if we have more than the top left corner, and discard if so
    if lrvb_cov.shape[0] == 2 * zs.shape[1]:
        lrvb_cov = lrvb_cov[: zs.shape[1], : zs.shape[1]]

    # TODO: Could use JAX here
    cov_chol = np.linalg.cholesky(lrvb_cov)
    draws = [opt_means + cov_chol @ cur_z for cur_z in zs]

    return np.array(draws)


def compute_lrvb_covariance_direct_method(
    opt_params, zs, hvp_fun, top_left_corner_only=True
):

    rel_hvp_fun = lambda b: hvp_fun(opt_params, zs, b)
    target_vecs = np.eye(opt_params.shape[0])

    hessian = np.stack([rel_hvp_fun(x) for x in target_vecs])

    # TODO: This could use JAX I guess.
    lrvb_cov_full = np.linalg.inv(hessian)

    if top_left_corner_only:
        n_rel = zs.shape[1]
        return lrvb_cov_full[:n_rel, :n_rel]
    else:
        return lrvb_cov_full


def compute_score_matrix(var_params, kl_est_and_grad_fun, zs):

    individual_grads = [
        kl_est_and_grad_fun(var_params, cur_z.reshape(1, -1))[1] for cur_z in zs
    ]
    grad_mat = np.array(individual_grads)

    return grad_mat


def compute_frequentist_covariance_estimate(
    var_params, kl_est_and_grad_fun, zs, lrvb_cov
):

    M = zs.shape[0]

    grad_mat = compute_score_matrix(var_params, kl_est_and_grad_fun, zs)

    expected_cov = (1 / M) * np.cov(grad_mat.T)
    expected_est = lrvb_cov @ expected_cov @ lrvb_cov

    return expected_est


def compute_preconditioner_from_var_params(var_params):

    means, log_sds = np.split(var_params, 2)
    inv_vars = np.concatenate([np.ones_like(log_sds), np.exp(-2*log_sds)])
    M = diags(inv_vars)

    return M


def compute_hessian_inv_column(var_params, index, hvp_fun, zs, preconditioner=None):

    oh_encoded = np.zeros_like(var_params)
    oh_encoded[index] = 1.0

    rel_hvp = lambda x: hvp_fun(var_params, zs, x)
    cg_result = cg_using_fun_scipy(rel_hvp, oh_encoded, preconditioner=preconditioner)
    success = cg_result[1] == 0

    return cg_result[0], success


def compute_single_frequentist_variance(index, var_params, dadvi_funs, zs, preconditioner=None):

    M = zs.shape[0]

    # TODO: These could be passed in instead
    rel_h, success = compute_hessian_inv_column(var_params, index, dadvi_funs.kl_est_hvp_fun, zs,
                                                preconditioner=preconditioner)
    score_mat = compute_score_matrix(var_params, dadvi_funs.kl_est_and_grad_fun, zs)

    # TODO: Check this is correct
    score_mat_means = score_mat.mean(axis=0, keepdims=True)
    centred_score_mat = score_mat - score_mat_means

    rel_estimate = np.einsum(
        "l,k,ml,mk->", rel_h, rel_h, centred_score_mat, centred_score_mat
    )

    return (1 / M**2) * rel_estimate
