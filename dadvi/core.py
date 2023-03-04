"""
Core computations for DADVI.
"""

from typing import NamedTuple, Callable, Optional, Dict

from scipy.sparse.linalg import LinearOperator

import numpy as np
from dadvi.optimization import optimize_with_hvp
from dadvi.utils import cg_using_fun_scipy


class DADVIFuns(NamedTuple):
    """
    This NamedTuple holds the functions required to run DADVI.

    Args:
    kl_est_and_grad_fun: Function of eta [variational parameters] and zs [draws].
        zs should have shape [M, D], where M is number of fixed draws and D is
        problem dimension. Returns a tuple whose first argument is the estimate
        of the KL divergence, and the second is its gradient w.r.t. eta.
    kl_est_hvp_fun: Function of eta, zs, and b, a vector to compute the hvp
        with. This should return a vector -- the result of the hvp with b.
    """

    kl_est_and_grad_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    kl_est_hvp_fun: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]


def find_dadvi_optimum(
    init_params: np.ndarray,
    zs: np.ndarray,
    dadvi_funs: DADVIFuns,
    opt_method: str = "trust-ncg",
    callback_fun: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """
    Optimises the DADVI objective.

    Args:
    init_params: The initial variational parameters to use. This should be a
        vector of length 2D, where D is the problem dimension. The first D
        entries specify the variational means, while the last D specify the log
        standard deviations.
    zs: The fixed draws to use in the optimisation. They must be of shape
        [M, D], where D is the problem dimension and M is the number of fixed
        draws.
    dadvi_funs: The objective to optimise. See the definition of DADVIFuns for
        more information. The kl_est_and_grad_fun is required for optimisation;
        the kl_est_hvp_fun is needed only for some optimisers.
    opt_method: The optimisation method to use. This must be one of the methods
        listed for scipy.optimize.minimize
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html].
        Defaults to trust-ncg, which requires the hvp to be available. For
        gradient-only optimisation, L-BFGS-B generally works well.
    callback_fun: If provided, this callback function is passed to
        scipy.optimize.minimize. See that function's documentation for more.
    verbose: If True, prints the progress of the optimisation by showing the
        value and gradient norm at each iteration of the optimizer.

    Returns:
    A dictionary with entries "opt_result", containing the results of running
    scipy.optimize.minimize, and "evaluation_count", containing the number of
    times the hvp and gradient functions were called.
    """

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
        verbose=verbose,
    )

    to_return = {
        "opt_result": opt_result,
        "evaluation_count": eval_count,
    }

    # If available, use hvp to check convergence
    if dadvi_funs.kl_est_hvp_fun is not None:
        problem_dimension = zs.shape[1]
        to_return["newton_step"] = compute_newton_step_vector(
            opt_result.x, zs, dadvi_funs
        )
        to_return["newton_step_norm"] = (
            np.linalg.norm(to_return["newton_step"]) / problem_dimension
        )

    return to_return


def get_dadvi_draws(var_params: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """
    Computes draws from the mean-field variational approximation given
    variational parameters and a matrix of fixed draws.

    Args:
        var_params: A vector of shape 2D, the first D entries specifying the
            means for the D model parameters, and the last D the log standard
            deviations.
        zs: A matrix of shape [N, D], containing the draws to use to sample the
            variational approximation.

    Returns:
    A matrix of shape [N, D] containing N draws from the variational
    approximation.
    """

    # TODO: Could use JAX here
    means, log_sds = np.split(var_params, 2)
    sds = np.exp(log_sds)

    draws = means.reshape(1, -1) + zs * sds.reshape(1, -1)

    return draws


def get_lrvb_draws(
    opt_means: np.ndarray, lrvb_cov: np.ndarray, zs: np.ndarray
) -> np.ndarray:
    """
    Computes draws from the LRVB approximation.

    Args:
        opt_means: A vector of shape D, where D is the number of model
            parameters, specifying the means of the variational approximation.
        lrvb_cov: A matrix of shape [D, D] containing the LRVB covariance of the
            D model parameters. Optionally, the [2D, 2D] inverted Hessian of the
            optimum can be passed; in this case, the top left corner will be
            used.
        zs: A matrix of shape [N, D] containing the standard normal draws that
            will be used to generate samples from the LRVB objective.

    Returns:
    A numpy array of shape [N, D] containing the N draws from the LRVB
    approximation to the objective.
    """

    # Check if we have more than the top left corner, and discard if so
    if lrvb_cov.shape[0] == 2 * zs.shape[1]:
        lrvb_cov = lrvb_cov[: zs.shape[1], : zs.shape[1]]

    # TODO: Could use JAX here
    cov_chol = np.linalg.cholesky(lrvb_cov)
    draws = [opt_means + cov_chol @ cur_z for cur_z in zs]

    return np.array(draws)


def compute_lrvb_covariance_direct_method(
    opt_params: np.ndarray,
    zs: np.ndarray,
    hvp_fun: Callable,
    top_left_corner_only: bool = True,
) -> np.ndarray:
    """
    Computes the LRVB covariance matrix by forming and inverting the Hessian.

    Args:
        opt_params: The variational parameters at the optimum. This is a vector
            of length 2D, the first D representing the the variational means,
            the second D the log standard deviations.
        zs: The fixed draws which were used to obtain the optimal parameters. A
            matrix of shape [M, D].
        hvp_fun: The hvp function of the DADVI objective, as defined in
            DADVIFuns.
        top_left_corner_only: If True, only the top left [D, D] corner of the
            full [2D, 2D] LRVB covariance matrix is returned.

    Returns:
    The [2D, 2D] LRVB covariance if top_left_corner_only = False, otherwise the
    [D, D] covariance corresponding to the model parameters.
    """

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


def compute_newton_step_vector(
    parameters: np.ndarray, zs: np.ndarray, dadvi_funs: DADVIFuns
) -> np.ndarray:
    """
    Computes the full vector of a Newton step. Useful for assessing convergence.
    """

    assert (
        dadvi_funs.kl_est_hvp_fun is not None
    ), "Can only compute Newton step norm if hvp is available!"

    cur_gradient = dadvi_funs.kl_est_and_grad_fun(parameters, zs)[1]
    hess_fun = lambda vector: dadvi_funs.kl_est_hvp_fun(parameters, zs, vector)

    # Compute H^{-1} g by CG:
    h_inv_g = cg_using_fun_scipy(hess_fun, cur_gradient, preconditioner=None)[0]

    return h_inv_g


def compute_newton_step_norm(
    parameters: np.ndarray, zs: np.ndarray, dadvi_funs: DADVIFuns
) -> float:
    """
    Computes the norm of a Newton step in optimisation. Helpful to check whether DADVI converged.
    """

    h_inv_g = compute_newton_step_vector(parameters, zs, dadvi_funs)
    h_inv_g_norm = np.linalg.norm(h_inv_g)

    return h_inv_g_norm


def compute_lrvb_covariance_cg(
    opt_params: np.ndarray,
    zs: np.ndarray,
    hvp_fun: Callable,
    top_left_corner_only: bool = True,
) -> np.ndarray:
    """
    Computes the LRVB covariance matrix by conjugate gradients.

    Args:
        opt_params: The variational parameters at the optimum. This is a vector
            of length 2D, the first D representing the the variational means,
            the second D the log standard deviations.
        zs: The fixed draws which were used to obtain the optimal parameters. A
            matrix of shape [M, D].
        hvp_fun: The hvp function of the DADVI objective, as defined in
            DADVIFuns.
        top_left_corner_only: If True, only the top left [D, D] corner of the
            full [2D, 2D] LRVB covariance matrix is returned.

    Returns:
    The [2D, 2D] LRVB covariance if top_left_corner_only = False, otherwise the
    [D, D] covariance corresponding to the model parameters.
    """

    preconditioner = compute_preconditioner_from_var_params(opt_params)

    D = zs.shape[1]

    if top_left_corner_only:
        relevant_indices = np.arange(D)
    else:
        relevant_indices = np.arange(2 * D)

    columns = list()

    for cur_index in relevant_indices:

        cur_column, _ = compute_hessian_inv_column(
            opt_params, cur_index, hvp_fun, zs, preconditioner=preconditioner
        )

        if top_left_corner_only:
            # We only need the first D entries
            columns.append(cur_column[:D])
        else:
            # We keep all of them
            columns.append(cur_column)

    full_result = np.stack(columns, axis=1)

    return full_result


def compute_score_matrix(var_params, kl_est_and_grad_fun, zs):
    """
    Computes the matrix of gradients w.r.t. the variational parameters at each
    of the fixed draws z.

    Args:
        var_params: The variational parameters to use to calculate the
            gradients.
        kl_est_and_grad_fun: The function returning the value and gradient of
            the DADVI KL estimate, as defined in DADVIFuns.
        zs: The [M, D] matrix of fixed draws.

    Returns:
    An [M, 2D] matrix containing the gradients evaluated at each fixed draw.
    """

    individual_grads = [
        kl_est_and_grad_fun(var_params, cur_z.reshape(1, -1))[1] for cur_z in zs
    ]
    grad_mat = np.array(individual_grads)

    return grad_mat


def compute_frequentist_covariance_estimate(
    var_params, kl_est_and_grad_fun, zs, lrvb_cov
):
    """
    Computes the frequentist covariance estimate, which estimates the error
    introduced by using fixed draws.

    Args:
        var_params: The variational parameter vector.
        kl_est_and_grad_fun: The function returning the value and gradient of
            the DADVI KL estimate, as defined in DADVIFuns.
        zs: The [M, D] matrix of fixed draws.
        lrvb_cov: The LRVB covariance matrix.

    Returns:
    The frequentist covariance matrix.
    """

    M = zs.shape[0]

    grad_mat = compute_score_matrix(var_params, kl_est_and_grad_fun, zs)

    # TODO: Do we need to discuss the denominator?
    expected_cov = (1 / (M - 1)) * np.cov(grad_mat.T)
    expected_est = lrvb_cov @ expected_cov @ lrvb_cov

    return expected_est


def compute_preconditioner_from_var_params(var_params):
    """
    Computes a diaagonal preconditioning matrix for CG from the variational
    parameters.
    """

    log_sds = np.split(var_params, 2)[1]
    variances = np.concatenate([np.exp(2 * log_sds), np.ones_like(log_sds)])

    def matvec(v):
        return variances * v

    op = LinearOperator(shape=(variances.shape[0], variances.shape[0]), matvec=matvec)

    return op


def compute_hessian_inv_column(var_params, index, hvp_fun, zs, preconditioner=None):
    """
    Computes a single column of the LRVB covariance matrix using CG.

    Args:
        var_params: The vector of variational parameters.
        index: The index of the column to compute.
        hvp_fun: The HVP of the DADVI objective as defined in DADVIFuns.
        zs: The matrix of fixed draws to use.
        preconditioner: An optional preconditioner for use with
            scipy.sparse.linalg.cg.

    Returns:
    A Tuple whose first element is the result of the CG iterations, and the
    second is a boolean success flag.
    """

    oh_encoded = np.zeros_like(var_params)
    oh_encoded[index] = 1.0

    rel_hvp = lambda x: hvp_fun(var_params, zs, x)
    cg_result = cg_using_fun_scipy(rel_hvp, oh_encoded, preconditioner=preconditioner)
    success = cg_result[1] == 0

    return cg_result[0], success


def compute_single_frequentist_variance(
    index, var_params, dadvi_funs, zs, preconditioner=None
):
    """
    Computes a single element on the diagonal of the frequentist covariance
    matrix using CG.

    Args:
        index: The index of the diagonal to compute.
        var_params: The vector of variational parameters.
        dadvi_funs: The DADVIFuns to use for the computation.
        zs: The matrix of fixed draws to use.
        preconditioner: The preconditioner to use in CG, defined as in
            scipy.sparse.linalg.cg.

    Returns:
    Element [index, index] of the frequentist covariance estimate.
    """

    M = zs.shape[0]

    rel_h, _ = compute_hessian_inv_column(
        var_params, index, dadvi_funs.kl_est_hvp_fun, zs, preconditioner=preconditioner
    )
    score_mat = compute_score_matrix(var_params, dadvi_funs.kl_est_and_grad_fun, zs)

    variance = compute_frequentist_covariance_using_score_mat(
        np.expand_dims(rel_h, axis=0), np.expand_dims(rel_h, axis=0), score_mat, M
    )[0, 0]

    return variance


def compute_frequentist_covariance_using_score_mat(
    hessian_cols1, hessian_cols2, score_mat, M
):
    """
    See "compute_single_frequentist_variance". Computes the same quantity, but
    using the relevant column of the Hessian and the score matrix rather than
    computing it first.
    """

    score_mat_means = score_mat.mean(axis=0, keepdims=True)
    centred_score_mat = score_mat - score_mat_means

    rel_estimate = np.einsum(
        "il,jk,ml,mk->ij",
        hessian_cols1,
        hessian_cols2,
        centred_score_mat,
        centred_score_mat,
    )

    return (1 / (M - 1) ** 2) * rel_estimate
