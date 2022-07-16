import os
import pymc3 as pm
import numpy as np


def fetch_tennis_model(start_year, sackmann_dir="./examples/tennis_atp"):
    from sklearn.preprocessing import LabelEncoder
    from jax_advi.data_utils.sackmann import get_data

    # TODO: Maybe download this if needed
    df = get_data(sackmann_dir)

    rel_df = df[df["tourney_date"].dt.year >= start_year]

    encoder = LabelEncoder()

    encoder.fit(
        rel_df["winner_name"].values.tolist() + rel_df["loser_name"].values.tolist()
    )

    names = encoder.classes_
    winner_ids = encoder.transform(rel_df["winner_name"])
    loser_ids = encoder.transform(rel_df["loser_name"])
    n_players = len(names)

    with pm.Model() as hierarchical_model:
        # Hyperpriors for group nodes
        sigma_player = pm.HalfNormal("sigma_player", 1.0)

        player_skills = pm.Normal(
            "player_skills", mu=0.0, sigma=sigma_player, shape=n_players
        )

        logit_skills = player_skills[winner_ids] - player_skills[loser_ids]

        # Data likelihood
        lik = pm.Bernoulli(
            "win_lik", logit_p=logit_skills, observed=np.ones(winner_ids.shape[0])
        )

    return hierarchical_model


def fetch_correlated_tennis_model(start_year, sackmann_dir="./examples/tennis_atp"):

    # TODO: Duplicated with the other tennis data. Maybe put some of this into a function.

    from jax_advi.data_utils.sackmann import get_data
    from sklearn.preprocessing import LabelEncoder

    df = get_data(sackmann_dir)

    rel_df = df[df["tourney_date"].dt.year >= start_year]

    encoder = LabelEncoder()

    encoder.fit(
        rel_df["winner_name"].values.tolist() + rel_df["loser_name"].values.tolist()
    )

    names = encoder.classes_
    winner_ids = encoder.transform(rel_df["winner_name"])
    loser_ids = encoder.transform(rel_df["loser_name"])

    n_players = len(names)

    surface_encoder = LabelEncoder()

    surface_encoder.fit(rel_df["surface"])
    surf_names = surface_encoder.classes_
    n_surf = len(surf_names)

    surface_ids = surface_encoder.transform(rel_df["surface"])

    with pm.Model() as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol",
            n=len(surf_names),
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True,
        )

        player_skills = pm.MvNormal(
            "player_skills", np.zeros(n_surf), chol=chol, shape=(n_players, n_surf)
        )

        logit_skills = (
            player_skills[winner_ids, surface_ids]
            - player_skills[loser_ids, surface_ids]
        )

        lik = pm.Bernoulli(
            "win_lik", logit_p=logit_skills, observed=np.ones(winner_ids.shape[0])
        )

    return model
