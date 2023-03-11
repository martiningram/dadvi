import json
import numpy as np
import pymc as pm
import pytensor
import jax
from pytensor.graph import Constant
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.shape import Reshape


@jax_funcify.register(Reshape)
def jax_funcify_Reshape(op, node, **kwargs):

    shape = node.inputs[1]
    if isinstance(shape, Constant):
        constant_shape = shape.data

        def reshape(x, _):
            return jax.numpy.reshape(x, constant_shape)

    else:

        def reshape(x, shape):
            return jax.numpy.reshape(x, shape)

    return reshape


def get_potus_model(stan_data_json):

    data = json.load(open(stan_data_json))

    np_data = {x: np.squeeze(np.array(y)) for x, y in data.items()}

    shapes = {
        "N_national_polls": int(np_data["N_national_polls"]),
        "N_state_polls": int(np_data["N_state_polls"]),
        "T": int(np_data["T"]),
        "S": int(np_data["S"]),
        "P": int(np_data["P"]),
        "M": int(np_data["M"]),
        "Pop": int(np_data["Pop"]),
    }

    national_cov_matrix_error_sd = np.sqrt(
        np.squeeze(
            np_data["state_weights"].reshape(1, -1)
            @ (np_data["state_covariance_0"] @ np_data["state_weights"].reshape(-1, 1))
        )
    )

    ss_cov_poll_bias = (
        np_data["state_covariance_0"]
        * (np_data["polling_bias_scale"] / national_cov_matrix_error_sd) ** 2
    )

    ss_cov_mu_b_T = (
        np_data["state_covariance_0"]
        * (np_data["mu_b_T_scale"] / national_cov_matrix_error_sd) ** 2
    )

    ss_cov_mu_b_walk = (
        np_data["state_covariance_0"]
        * (np_data["random_walk_scale"] / national_cov_matrix_error_sd) ** 2
    )

    cholesky_ss_cov_poll_bias = np.linalg.cholesky(ss_cov_poll_bias)
    cholesky_ss_cov_mu_b_T = np.linalg.cholesky(ss_cov_mu_b_T)
    cholesky_ss_cov_mu_b_walk = np.linalg.cholesky(ss_cov_mu_b_walk)

    i, j = np.indices((np_data["T"], np_data["T"]))

    mask = np.tril(np.ones((np_data["T"], np_data["T"])))

    with pm.Model() as m2:

        normal_dist = pm.Normal.dist(mu=0.7, sigma=0.1)

        # BoundedNormalZeroOne = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
        raw_polling_bias = pm.Normal(
            "raw_polling_bias", mu=0.0, sigma=1.0, shape=(shapes["S"])
        )

        raw_mu_b_T = pm.Normal("raw_mu_b_T", mu=0.0, sigma=1.0, shape=(shapes["S"]))
        raw_mu_b = pm.Normal(
            "raw_mu_b", mu=0.0, sigma=1.0, shape=(shapes["S"], shapes["T"])
        )
        raw_mu_c = pm.Normal("raw_mu_c", mu=0.0, sigma=1.0, shape=(shapes["P"]))
        raw_mu_m = pm.Normal("raw_mu_m", mu=0.0, sigma=1.0, shape=(shapes["M"]))
        raw_mu_pop = pm.Normal("raw_mu_pop", mu=0.0, sigma=1.0, shape=(shapes["Pop"]))

        # This has offset multiplier syntax in Stan, but ignore for now.
        mu_e_bias = pm.Normal("mu_e_bias", mu=0.0, sigma=0.02)

        # This may be an issue?
        # rho_e_bias = BoundedNormalZeroOne("rho_e_bias", mu=0.7, sigma=0.1)
        rho_e_bias = pm.Bound("rho_e_bias", normal_dist, lower=0.0, upper=1.0)

        raw_e_bias = pm.Normal("raw_e_bias", mu=0.0, sigma=1.0, shape=(shapes["T"]))
        raw_measure_noise_national = pm.Normal(
            "raw_measure_noise_national",
            mu=0.0,
            sigma=1.0,
            shape=(shapes["N_national_polls"]),
        )
        raw_measure_noise_state = pm.Normal(
            "raw_measure_noise_state",
            mu=0.0,
            sigma=1.0,
            shape=(shapes["N_state_polls"]),
        )

        polling_bias = pm.Deterministic(
            "polling_bias",
            pytensor.tensor.dot(cholesky_ss_cov_poll_bias, raw_polling_bias),
        )
        national_polling_bias_average = pm.Deterministic(
            "national_polling_bias_average",
            pm.math.sum(polling_bias * np_data["state_weights"]),
        )

        mu_b_final = (
            pm.math.dot(cholesky_ss_cov_mu_b_T, raw_mu_b_T) + np_data["mu_b_prior"]
        )

        # Innovations
        innovs = pm.math.matrix_dot(cholesky_ss_cov_mu_b_walk, raw_mu_b[:, :-1])

        # Reverse these (?)
        innovs = pytensor.tensor.transpose(pytensor.tensor.transpose(innovs)[::-1])

        # Tack on the "first" one:
        together = pm.math.concatenate(
            [pytensor.tensor.reshape(mu_b_final, (-1, 1)), innovs], axis=1
        )

        # Compute the cumulative sums:
        cumsums = pytensor.tensor.cumsum(together, axis=1)

        # To be [time, states]
        transposed = pytensor.tensor.transpose(cumsums)

        mu_b = pm.Deterministic("mu_b", pytensor.tensor.transpose(transposed[::-1]))

        national_mu_b_average = pm.Deterministic(
            "national_mu_b_average",
            pm.math.matrix_dot(
                pytensor.tensor.transpose(mu_b),
                np_data["state_weights"].reshape((-1, 1)),
            ),
        )[:, 0]

        mu_c = pm.Deterministic("mu_c", raw_mu_c * np_data["sigma_c"])
        mu_m = pm.Deterministic("mu_m", raw_mu_m * np_data["sigma_m"])
        mu_pop = pm.Deterministic("mu_pop", raw_mu_pop * np_data["sigma_pop"])

        # Matrix version:

        # e_bias_init = raw_e_bias[0] * np_data['sigma_e_bias']
        sigma_rho = pm.math.sqrt(1 - (rho_e_bias**2)) * np_data["sigma_e_bias"]
        sigma_vec = pm.math.concatenate(
            [
                [np_data["sigma_e_bias"]],
                pytensor.tensor.repeat(sigma_rho, np_data["T"] - 1),
            ]
        )
        mus = pm.math.concatenate(
            [
                [0.0],
                pytensor.tensor.repeat(mu_e_bias * (1 - rho_e_bias), shapes["T"] - 1),
            ]
        )

        A_inv = mask * (rho_e_bias ** (np.abs(i - j)))

        e_bias = pm.Deterministic(
            "e_bias", pm.math.matrix_dot(A_inv, sigma_vec * raw_e_bias + mus)
        )

        # Minus ones shenanigans required for different indexing
        logit_pi_democrat_state = (
            mu_b[np_data["state"] - 1, np_data["day_state"] - 1]
            + mu_c[np_data["poll_state"] - 1]
            + mu_m[np_data["poll_mode_state"] - 1]
            + mu_pop[np_data["poll_pop_state"] - 1]
            + np_data["unadjusted_state"] * e_bias[np_data["day_state"] - 1]
            + raw_measure_noise_state * np_data["sigma_measure_noise_state"]
            + polling_bias[np_data["state"] - 1]
        )

        logit_pi_democrat_state = pm.Deterministic(
            "logit_pi_democrat_state", logit_pi_democrat_state
        )

        logit_pi_democrat_national = (
            national_mu_b_average[np_data["day_national"] - 1]
            + mu_c[np_data["poll_national"] - 1]
            + mu_m[np_data["poll_mode_national"] - 1]
            + mu_pop[np_data["poll_pop_national"] - 1]
            + np_data["unadjusted_national"] * e_bias[np_data["day_national"] - 1]
            + raw_measure_noise_national * np_data["sigma_measure_noise_national"]
            + national_polling_bias_average
        )

        logit_pi_democrat_national = pm.Deterministic(
            "logit_pi_democrat_national", logit_pi_democrat_national
        )

        prob_state = pytensor.tensor.sigmoid(logit_pi_democrat_state)
        prob_nat = pytensor.tensor.sigmoid(logit_pi_democrat_national)

        state_lik = pm.Binomial(
            "state_lik",
            n=np_data["n_two_share_state"],
            p=prob_state,
            observed=np_data["n_democrat_state"],
        )
        national_lik = pm.Binomial(
            "nat_lik",
            n=np_data["n_two_share_national"],
            p=prob_nat,
            observed=np_data["n_democrat_national"],
        )

    return m2
