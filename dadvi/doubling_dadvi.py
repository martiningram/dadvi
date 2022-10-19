"""
Functions for running DADVI and doubling the number of fixed draws until the frequentist variance
resulting from the fixed number of draws is small compared to the posterior variances.
"""
import numpy as np
from dadvi.core import (
    compute_frequentist_covariance_estimate,
    compute_lrvb_covariance_direct_method,
    find_dadvi_optimum,
)


def fit_dadvi_and_estimate_covariances(init_params, zs, dadvi_funs, **kwargs):
    """
    Finds the DADVI optimum for the given fixed draws and initial parameters, then
    estimates the LRVB covariance and frequentist covariances.
    """

    opt = find_dadvi_optimum(
        init_params=init_params, zs=zs, dadvi_funs=dadvi_funs, **kwargs
    )

    opt_var_params = opt["opt_result"]["x"]

    # Compute LRVB covariance estimate and use it to estimate the frequentist covariance matrix
    lrvb_cov = compute_lrvb_covariance_direct_method(
        opt_var_params, zs, dadvi_funs.kl_est_hvp_fun, top_left_corner_only=False
    )
    freq_cov = compute_frequentist_covariance_estimate(
        opt_var_params, dadvi_funs.kl_est_and_grad_fun, zs, lrvb_cov
    )

    # Compute the frequentist standard deviation of the means
    frequentist_variances = np.diag(freq_cov)
    frequentist_mean_variances = np.split(frequentist_variances, 2)[0]
    frequentist_mean_sds = np.sqrt(frequentist_mean_variances)

    # Compute the LRVB standard deviations
    variational_sds = np.sqrt(np.split(np.diag(lrvb_cov), 2)[0])

    return {
        "optimisation_result": opt,
        "lrvb_covariance": lrvb_cov,
        "frequentist_covariance": freq_cov,
        "frequentist_mean_sds": frequentist_mean_sds,
        "variational_sds": variational_sds,
    }


def optimise_dadvi_by_doubling(
    init_params,
    dadvi_funs,
    start_m_power=2,
    max_m_power=7,
    seed=2,
    max_freq_to_posterior_ratio=0.5,
    **kwargs
):
    """
    Repeatedly doubles the number of fixed draws until the ratio of the frequentist standard deviations
    to the variational standard deviations drops below the specified ratio.
    """

    assert start_m_power < max_m_power, "Minimum power for M must be smaller than maximum power!"

    np.random.seed(seed)

    n_model_params = init_params.shape[0] // 2

    for cur_m_power in range(start_m_power, max_m_power + 1):

        cur_m = 2**cur_m_power

        # Make current fixed draws
        zs = np.random.randn(cur_m, n_model_params)

        # Run DADVI
        dadvi_result = fit_dadvi_and_estimate_covariances(
            init_params, zs, dadvi_funs, **kwargs
        )

        # Compute the ratios
        freq_to_posterior_ratios = (
            dadvi_result["frequentist_mean_sds"] / dadvi_result["variational_sds"]
        )
        freq_to_posterior_ratio = max(freq_to_posterior_ratios)

        ratio_is_ok = freq_to_posterior_ratio < max_freq_to_posterior_ratio

        if ratio_is_ok:
            break

    return {
        "dadvi_result": dadvi_result,
        "zs": zs,
        "ratio": freq_to_posterior_ratio,
        "ratio_is_ok": ratio_is_ok,
        "M": cur_m,
    }
