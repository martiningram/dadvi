import numpy as np
import pandas as pd
from functools import partial
import pymc as pm
import pytensor
from pytensor.tensor.extra_ops import bincount
import pickle
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


def flatten_cells(cell_ids, n_species, n_cells, y):

    cell_ids_tiled = np.tile(cell_ids, (n_species, 1))
    add_per_row = n_cells
    to_add = np.arange(n_species) * add_per_row

    new_ids = cell_ids_tiled + to_add.reshape(-1, 1)
    new_ids_flat = new_ids.reshape(-1)

    y_flat = y.T.reshape(-1)

    return {"cell_ids_flat": new_ids_flat, "n_cells": n_cells, "y_flat": y_flat}


def checklist_likelihood_from_combined(
    env_and_obs_logits, y, cell_nums, n_cells, obs_per_cell, n_env_logits
):

    rel_env_logits = env_and_obs_logits[:n_env_logits]
    rel_obs_logits = env_and_obs_logits[n_env_logits:]

    return checklist_likelihood(
        rel_env_logits, rel_obs_logits, y, cell_nums, n_cells, obs_per_cell
    )


def checklist_likelihood(y, env_logits, obs_logits, cell_nums, n_cells, obs_per_cell):

    link_fun = lambda x: pm.logcdf(pm.Logistic.dist(), x)

    log_prob_pres = link_fun(env_logits)
    log_prob_abs = link_fun(-env_logits)

    log_prob_miss_if_pres = link_fun(-obs_logits)
    log_prob_obs_if_pres = link_fun(obs_logits)

    rel_log_probs = pm.math.where(
        pytensor.tensor.eq(y, 1), log_prob_obs_if_pres, log_prob_miss_if_pres
    )

    summed_liks = bincount(cell_nums, weights=rel_log_probs, minlength=n_cells)

    lik_term_1 = log_prob_abs
    lik_term_2 = log_prob_pres + summed_liks

    lik_if_at_least_one_obs = lik_term_2
    lik_if_all_missing = pm.math.logsumexp(
        pm.math.stack([lik_term_1, lik_term_2], axis=1), axis=1, keepdims=False
    )

    log_lik = pm.math.where(
        pytensor.tensor.eq(obs_per_cell, 0), lik_if_all_missing, lik_if_at_least_one_obs
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

    with pm.Model() as m:

        # Some seemingly unnecessary transposes here to make structured vb
        # easier down the road (want species as first dimension)
        w_env = pytensor.tensor.transpose(
            pm.Normal(
                "w_env",
                mu=0.0,
                sigma=1.0,
                shape=(n_species, data["X_env_mat"].shape[1]),
            )
        )
        intercept = pytensor.tensor.transpose(
            pm.Normal("intercept", mu=0.0, sigma=10.0, shape=(n_species, 1))
        )

        env_logits = pytensor.tensor.transpose(
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
            pytensor.tensor.transpose(w_obs_raw) * obs_prior_sds + obs_prior_means,
        )

        obs_logits = pytensor.tensor.transpose(
            pm.math.matrix_dot(data["X_obs_mat"], w_obs)
        )

        env_logits_flat = pytensor.tensor.reshape(env_logits, (-1,))
        obs_logits_flat = pytensor.tensor.reshape(obs_logits, (-1,))

        n_env_logits = data["X_env_mat"].shape[0] * n_species

        cur_lik = partial(
            checklist_likelihood,
            cell_nums=flat_version["cell_ids_flat"],
            n_cells=flat_version["n_cells"],
            obs_per_cell=obs_per_cell,
        )

        pm.DensityDist(
            f"obs_lik",
            env_logits_flat,
            obs_logits_flat,
            logp=cur_lik,
            observed=flat_version["y_flat"],
        )

    return m


def get_occ_det_model_from_pickle(pickle_file, n_species=32, n_checklists=2000):

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    return get_occ_det_model_from_data(data, n_species, n_checklists)
