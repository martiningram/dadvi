import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
from dadvi.core import find_dadvi_optimum
from dadvi.jax import build_dadvi_funs


def test__when_run_on_uncorrelated_gaussian__then_should_recover_parameters():
    np.random.seed(2)

    test_means = jnp.array([1.0, 2.0])
    test_sds = jnp.array([3.0, 4.0])

    n_params = test_means.shape[0]
    n_draws = 50000

    @jit
    def target_log_density(x):
        return jnp.sum(norm.logpdf(x, loc=test_means, scale=test_sds))

    dadvi_funs = build_dadvi_funs(target_log_density)
    zs = np.random.randn(n_draws, n_params)
    init_params = np.zeros(2 * n_params)

    optimum = find_dadvi_optimum(init_params=init_params, zs=zs, dadvi_funs=dadvi_funs)

    assert optimum["opt_result"].success

    opt_means = optimum["opt_result"].x[:n_params]
    opt_sds = np.exp(optimum["opt_result"].x[n_params:])

    assert np.allclose(opt_means, test_means, rtol=5e-2)
    assert np.allclose(opt_sds, test_sds, rtol=5e-2)
