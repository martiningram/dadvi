from typing import Callable, Dict, Tuple, Optional

import numpy as np
from jax import vmap

from dadvi.jax import (
    build_dadvi_funs,
    compute_posterior_mean_and_sd_using_cg_delta_method,
)
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.core import find_dadvi_optimum, get_dadvi_draws, DADVIFuns
from dadvi.utils import opt_callback_fun


class DADVIResult:
    def __init__(
        self,
        fixed_draws: np.ndarray,
        var_params: np.ndarray,
        unflattening_fun: Callable[[np.ndarray], Dict[str, np.ndarray]],
        dadvi_funs: DADVIFuns,
    ):

        self.fixed_draws = fixed_draws
        self.var_params = var_params
        self.unflattening_fun = unflattening_fun
        self.dadvi_funs = dadvi_funs
        self.n_params = self.fixed_draws.shape[1]

    def get_posterior_means(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with posterior means for all parameters.
        """

        means = np.split(self.var_params, 2)[0]
        return self.unflattening_fun(means)

    def get_posterior_standard_deviations_mean_field(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with posterior standard deviations (not LRVB-corrected, but mean field).
        """

        log_sds = np.split(self.var_params, 2)[1]
        sds = np.exp(log_sds)
        return self.unflattening_fun(sds)

    def get_posterior_draws_mean_field(
        self,
        n_draws: int = 1000,
        seed: int = 2,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with draws from the posterior.
        """

        np.random.seed(seed)
        z = np.random.randn(n_draws, self.n_params)
        dadvi_draws_flat = get_dadvi_draws(self.var_params, z)
        dadvi_dict = vmap(self.unflattening_fun)(dadvi_draws_flat)

        return dadvi_dict

    def get_mean_and_sd_of_scalar_valued_function(
        self, fun_to_compute
    ) -> Tuple[float, float]:
        """
        Given a function that computes a scalar-valued output from a dictionary of parameters,
        returns the posterior mean and variance of this result using the delta method and
        conjugate gradients.
        """

        return compute_posterior_mean_and_sd_using_cg_delta_method(
            fun_to_compute,
            self.var_params,
            self.fixed_draws,
            self.dadvi_funs,
            self.unflattening_fun,
        )


def fit_pymc_dadvi_with_jax(pymc_model, num_fixed_draws=30):

    jax_funs = get_jax_functions_from_pymc(pymc_model)
    dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

    opt_callback_fun.opt_sequence = []

    init_means = np.zeros(jax_funs["n_params"])
    init_log_vars = np.zeros(jax_funs["n_params"]) - 3
    init_var_params = np.concatenate([init_means, init_log_vars])
    zs = np.random.randn(num_fixed_draws, jax_funs["n_params"])
    opt = find_dadvi_optimum(
        init_params=init_var_params,
        zs=zs,
        dadvi_funs=dadvi_funs,
        verbose=True,
        callback_fun=opt_callback_fun,
    )

    return DADVIResult(
        fixed_draws=zs,
        var_params=opt["opt_result"].x,
        unflattening_fun=jax_funs["unflattening_fun"],
        dadvi_funs=dadvi_funs,
    )
