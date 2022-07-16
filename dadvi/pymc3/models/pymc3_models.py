import pymc3 as pm
import pandas as pd
import theano


def fetch_radon_model():

    data = pd.read_csv(pm.get_data("radon.csv"))
    data["log_radon"] = data["log_radon"].astype(theano.config.floatX)
    county_names = data.county.unique()
    county_idx = data.county_code.values.astype("int32")

    n_counties = len(data.county.unique())

    with pm.Model() as hierarchical_model:
        # Hyperpriors for group nodes
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=100.0)
        sigma_a = pm.HalfNormal("sigma_a", 5.0)
        mu_b = pm.Normal("mu_b", mu=0.0, sigma=100.0)
        sigma_b = pm.HalfNormal("sigma_b", 5.0)

        # Intercept for each county, distributed around group mean mu_a
        # Above we just set mu and sd to a fixed value while here we
        # plug in a common group distribution for all a and b (which are
        # vectors of length n_counties).
        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_counties)
        # Intercept for each county, distributed around group mean mu_a
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_counties)

        # Model error
        eps = pm.HalfCauchy("eps", 5.0)

        radon_est = a[county_idx] + b[county_idx] * data.floor.values

        # Data likelihood
        radon_like = pm.Normal(
            "radon_like", mu=radon_est, sigma=eps, observed=data.log_radon
        )

    return hierarchical_model
