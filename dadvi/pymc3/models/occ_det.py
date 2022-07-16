import numpy as np
import patsy
from jax_advi.utils.patsy import remove_intercept_column, create_formula
from sklearn.preprocessing import StandardScaler
import pandas as pd
from functools import partial
import pymc3 as pm
import theano
from jax_advi.utils.pymc3 import log_sigmoid
from theano.tensor.extra_ops import bincount


def load_and_prepare_data(n_species=32, n_checklists=2000, min_obs=10, seed=2):
    from occu_py.checklist_dataset import (
        load_ebird_dataset_using_env_var,
        random_checklist_subset,
    )

    np.random.seed(seed)

    # TODO: Probably will just want to put the subset in a csv

    ebird = load_ebird_dataset_using_env_var()
    env_data = ebird["train"].X_env
    checklist_data = ebird["train"].X_obs

    obs_formula = "protocol_type + daytimes_alt + log_duration_z + dominant_land_cover"

    subset = random_checklist_subset(
        checklist_data.shape[0], ebird["train"].env_cell_ids, n_checklists, seed=seed
    )

    X_env = env_data.iloc[subset["env_cell_indices"]].copy()
    X_obs = checklist_data.iloc[subset["checklist_indices"]].copy()

    y = ebird["train"].y_obs.iloc[subset["checklist_indices"]]
    y = y[y.columns[(y.sum() >= min_obs)]]

    bird_choice = np.random.choice(y.columns, size=n_species, replace=False)

    y_df = y[bird_choice].copy()
    y = y[bird_choice].values.astype(int)

    n_species = y.shape[1]

    log_duration_mean = X_obs["log_duration"].mean()
    log_duration_sd = X_obs["log_duration"].std()

    X_obs["log_duration_z"] = (
        X_obs["log_duration"] - log_duration_mean
    ) / log_duration_sd

    bio_covs = [x for x in X_env.columns if "bio" in x]

    env_formula = create_formula(
        bio_covs,
        main_effects=True,
        quadratic_effects=True,
        interactions=False,
        intercept=True,
    )

    to_add = [x for x in X_env.columns if x.startswith("has_")]

    combined = "+".join(to_add)

    env_formula = env_formula + "+" + combined

    env_scaler = StandardScaler()

    bio_scaled = pd.DataFrame(
        env_scaler.fit_transform(X_env[bio_covs]), index=X_env.index, columns=bio_covs
    )

    X_env_to_use = pd.concat([bio_scaled, X_env[to_add]], axis=1)

    X_env_mat = patsy.dmatrix(env_formula, X_env_to_use)
    env_design_info = X_env_mat.design_info
    X_obs_mat = patsy.dmatrix(obs_formula, X_obs)
    obs_design_info = X_obs_mat.design_info
    X_env_mat = remove_intercept_column(X_env_mat, X_env_mat.design_info)

    n_cells = np.max(np.unique(subset["checklist_cell_ids"])) + 1
    n_checklists = subset["checklist_indices"].shape[0]
    cell_ids = subset["checklist_cell_ids"]

    return {
        "X_env_mat": X_env_mat,
        "X_obs_mat": X_obs_mat,
        "env_formula": env_formula,
        "obs_formula": obs_formula,
        "X_env_df_scaled": X_env_to_use,
        "X_obs_df": X_obs,
        "n_cells": n_cells,
        "n_checklists": n_checklists,
        "cell_ids": cell_ids,
        "y": y,
        "y_df": y_df,
        "env_scaler": env_scaler,
        "env_design_info": env_design_info,
        "log_duration_mean": log_duration_mean,
        "log_duration_sd": log_duration_sd,
        "obs_design_info": obs_design_info,
    }


def flatten_cells(cell_ids, n_species, n_cells, y):

    cell_ids_tiled = np.tile(cell_ids, (n_species, 1))
    add_per_row = n_cells
    to_add = np.arange(n_species) * add_per_row

    new_ids = cell_ids_tiled + to_add.reshape(-1, 1)
    new_ids_flat = new_ids.reshape(-1)
    new_n_cells = np.max(new_ids) + 1

    y_flat = y.T.reshape(-1)

    return {"cell_ids_flat": new_ids_flat, "n_cells": n_cells, "y_flat": y_flat}


def checklist_likelihood(env_logits, obs_logits, y, cell_nums, n_cells, obs_per_cell):

    # link_fun = log_sigmoid
    # link_fun = pm.Normal.dist(mu=0, sigma=1).logcdf
    link_fun = pm.Logistic.dist().logcdf

    log_prob_pres = link_fun(env_logits)
    log_prob_abs = link_fun(-env_logits)

    log_prob_miss_if_pres = link_fun(-obs_logits)
    log_prob_obs_if_pres = link_fun(obs_logits)

    rel_log_probs = pm.math.where(
        theano.tensor.eq(y, 1), log_prob_obs_if_pres, log_prob_miss_if_pres
    )

    summed_liks = bincount(cell_nums, weights=rel_log_probs, minlength=n_cells)

    lik_term_1 = log_prob_abs
    lik_term_2 = log_prob_pres + summed_liks

    lik_if_at_least_one_obs = lik_term_2
    lik_if_all_missing = pm.math.logsumexp(
        pm.math.stack([lik_term_1, lik_term_2], axis=1), axis=1, keepdims=False
    )

    log_lik = pm.math.where(
        theano.tensor.eq(obs_per_cell, 0), lik_if_all_missing, lik_if_at_least_one_obs
    )

    return log_lik.sum()


def get_occ_det_model_from_data(data, n_species, n_checklists):

    flat_version = flatten_cells(
        data["cell_ids"], n_species, data["n_cells"], data["y"]
    )

    obs_per_cell = np.bincount(
        flat_version["cell_ids_flat"],
        weights=flat_version["y_flat"],
        minlength=flat_version["n_cells"],
    )

    cur_lik = partial(
        checklist_likelihood,
        cell_nums=flat_version["cell_ids_flat"],
        n_cells=flat_version["n_cells"],
        obs_per_cell=obs_per_cell,
        y=flat_version["y_flat"],
    )

    with pm.Model() as m:

        # Some seemingly unnecessary transposes here to make structured vb
        # easier down the road (want species as first dimension)
        w_env = theano.tensor.transpose(
            pm.Normal(
                "w_env",
                mu=0.0,
                sigma=1.0,
                shape=(n_species, data["X_env_mat"].shape[1]),
            )
        )
        intercept = theano.tensor.transpose(
            pm.Normal("intercept", mu=0.0, sigma=10.0, shape=(n_species, 1))
        )

        env_logits = theano.tensor.transpose(
            pm.math.matrix_dot(data["X_env_mat"], w_env) + intercept
        )

        obs_prior_means = pm.Normal(
            "w_prior_mean", mu=0.0, sigma=1.0, shape=(data["X_obs_mat"].shape[1], 1)
        )
        obs_prior_sds = pm.HalfNormal(
            "w_prior_sd", sigma=1.0, shape=(data["X_obs_mat"].shape[1], 1)
        )

        # Multi-species here.
        w_obs_raw = pm.Normal(
            "w_obs_raw",
            mu=0.0,
            sigma=1.0,
            # shape=(data["X_obs_mat"].shape[1], n_species),
            shape=(n_species, data["X_obs_mat"].shape[1]),
        )
        w_obs = pm.Deterministic(
            "w_obs",
            theano.tensor.transpose(w_obs_raw) * obs_prior_sds + obs_prior_means,
        )

        obs_logits = theano.tensor.transpose(
            pm.math.matrix_dot(data["X_obs_mat"], w_obs)
        )

        env_logits_flat = theano.tensor.reshape(env_logits, (-1,))
        obs_logits_flat = theano.tensor.reshape(obs_logits, (-1,))

        pm.DensityDist(
            f"obs_lik",
            cur_lik,
            observed={
                "env_logits": env_logits_flat,
                "obs_logits": obs_logits_flat,
            },
        )

    return m


def get_occ_det_model(n_species=32, n_checklists=2000, min_obs=10, seed=2):

    data = load_and_prepare_data(n_species, n_checklists, min_obs, seed)

    return get_occ_det_model_from_data(data, n_species, n_checklists)


def get_occ_det_model_from_pickle(pickle_file, n_species=32, n_checklists=2000):

    # TODO: Infer species and checklists from pickle instead

    import pickle

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    return get_occ_det_model_from_data(data, n_species, n_checklists)
