import pymc as pm
import numpy as np
from dadvi.pymc.jax_api import fit_pymc_dadvi_with_jax


def test__when_given_uncorrelated_normal_in_pymc__then_should_recover_parameters():
    means = np.array([0.0, 2.0])
    sds = np.array([1.0, 2.0])

    with pm.Model() as m:
        _ = pm.Normal("mu", mu=means, sigma=sds)

    dadvi_result = fit_pymc_dadvi_with_jax(m, num_fixed_draws=10000, seed=2)

    assert np.sum((means - dadvi_result.get_posterior_means()["mu"]) ** 2) < 1e-3
    assert (
        np.sum(
            (sds - dadvi_result.get_posterior_standard_deviations_mean_field()["mu"])
            ** 2
        )
        < 1e-3
    )
