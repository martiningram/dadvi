import json
import numpy as np
import theano
import pymc3 as pm


def load_microcredit_model(data_json_path):

    loaded = json.load(open(data_json_path))

    np_data = {x: np.squeeze(np.array(y)) for x, y in loaded.items()}

    K = np_data["K"]
    M = np_data["M"]
    P = np_data["P"]

    with pm.Model() as m:

        BoundedCauchy = pm.Bound(pm.Cauchy, lower=0.0)

        mu = pm.Normal("mu", mu=0.0, sigma=100.0, shape=2)
        sd_mu = BoundedCauchy("sd_mu", alpha=0.0, beta=2.0, shape=2)

        tau = pm.Normal("tau", mu=0.0, sigma=100.0, shape=2)
        sd_tau = BoundedCauchy("sd_tau", alpha=0.0, beta=2.0, shape=2)

        sigma_control = pm.Normal("sigma_control", mu=0.0, sigma=100.0, shape=2)
        sd_sigma_control = BoundedCauchy(
            "sd_sigma_control", alpha=0.0, beta=2.0, shape=2
        )

        sigma_TE = pm.Normal("sigma_TE", mu=0.0, sigma=100.0, shape=2)
        sd_sigma_TE = BoundedCauchy("sd_sigma_TE", alpha=0.0, beta=2.0, shape=2)

        mu_k = pm.Normal("mu_k", mu=mu, sigma=sd_mu, shape=(K, 2))
        tau_k = pm.Normal("tau_k", mu=tau, sigma=sd_tau, shape=(K, 2))
        sigma_control_k = pm.Normal(
            "sigma_control_k", mu=sigma_control, sigma=sd_sigma_control, shape=(K, 2)
        )
        sigma_TE_k = pm.Normal(
            "sigma_TE_k", mu=sigma_TE, sigma=sd_sigma_TE, shape=(K, 2)
        )

        sigma = BoundedCauchy("sigma", alpha=0.0, beta=2.0, shape=(M, P))
        beta_k_raw = pm.Normal("beta_k_raw", mu=0.0, sigma=1.0, shape=(K, M, P))

        beta = pm.Normal("beta", mu=0.0, sigma=5.0, shape=(M - 1, P))
        beta_full = pm.Deterministic(
            "beta_full", pm.math.concatenate([beta, np.zeros((1, P))])
        )

        beta_k = pm.Deterministic("beta_k", beta_full + sigma * beta_k_raw)

        obs_logits = pm.math.sum(
            beta_k[np_data["site"] - 1] * np.expand_dims(np_data["x"], axis=1), axis=-1
        )

        obs_probs = theano.tensor.nnet.softmax(obs_logits)

        cat_lik = pm.Categorical("cat_lik", p=obs_probs, observed=np_data["cat"] - 1)

        neg_mean = (
            mu_k[np_data["site_neg"] - 1, 0]
            + tau_k[np_data["site_neg"] - 1, 0] * np_data["treatment_neg"]
        )
        neg_sd = pm.math.exp(
            sigma_control_k[np_data["site_neg"] - 1, 0]
            + sigma_TE_k[np_data["site_neg"] - 1, 0] * np_data["treatment_neg"]
        )

        pos_mean = (
            mu_k[np_data["site_pos"] - 1, 1]
            + tau_k[np_data["site_pos"] - 1, 1] * np_data["treatment_pos"]
        )
        pos_sd = pm.math.exp(
            sigma_control_k[np_data["site_pos"] - 1, 1]
            + sigma_TE_k[np_data["site_pos"] - 1, 1] * np_data["treatment_pos"]
        )

        # TODO: Make sure parameterisation is what's expected. Docs are a bit confusing.
        y_neg_lik = pm.Lognormal(
            "y_neg_lik", mu=neg_mean, sigma=neg_sd, observed=np_data["y_neg"]
        )
        y_pos_lik = pm.Lognormal(
            "y_pos_lik", mu=pos_mean, sigma=pos_sd, observed=np_data["y_pos"]
        )

    return m
