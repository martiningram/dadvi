import numpy as np
import pandas as pd
from config import (
    ARM_CONFIG_CSV_PATH,
    ARM_JSON_PATH,
    MICROCREDIT_JSON_PATH,
    OCC_DET_PICKLE_PATH,
    POTUS_JSON_PATH,
    SACKMANN_DIR,
)
from dadvi.pymc.models.arm import load_arm_model, add_bambi_family
from dadvi.pymc.models.microcredit import load_microcredit_model
from dadvi.pymc.models.occ_det import get_occ_det_model_from_pickle
from dadvi.pymc.models.potus import get_potus_model
from dadvi.pymc.models.tennis import fetch_tennis_model
import pymc as pm


NON_ARM_MODELS = ["microcredit", "occ_det", "potus", "tennis"]


def load_model_by_name(model_name):

    if model_name == "microcredit":
        model = load_microcredit_model(MICROCREDIT_JSON_PATH)

    elif model_name == "occ_det":
        model = get_occ_det_model_from_pickle(OCC_DET_PICKLE_PATH)

    elif model_name == "potus":
        model = get_potus_model(POTUS_JSON_PATH)

    elif model_name == "tennis":
        model = fetch_tennis_model(1969, sackmann_dir=SACKMANN_DIR)

    else:
        df = pd.read_csv(ARM_CONFIG_CSV_PATH)
        df = add_bambi_family(df)

        # Should check for duplicates here
        rel_row = df[df["model_name"] == model_name].iloc[0]

        model = load_arm_model(rel_row, ARM_JSON_PATH)["pymc_model"]

    return model


def estimate_kl_fresh_draws(dadvi_funs, var_params, n_draws=1000):

    n_params = len(var_params) // 2

    cur_z = np.random.randn(n_draws, n_params)
    return dadvi_funs.kl_est_and_grad_fun(var_params, cur_z)[0]


def arviz_to_draw_dict(az_trace):
    # Converts an arviz trace to a dict of str -> np.ndarray

    dict_version = dict(az_trace.posterior.data_vars.variables)

    return {x: y.values for x, y in dict_version.items()}


def fit_pymc_sadvi(
    m, n_draws=1000, n_steps=100000, method="advi", convergence_crit="default"
):

    assert method in ["advi", "fullrank_advi"]

    if convergence_crit is None:
        extra_args = {}
    elif convergence_crit == "default":
        extra_args = {"callbacks": [pm.callbacks.CheckParametersConvergence()]}
    elif convergence_crit == "absdiff":
        extra_args = {
            "callbacks": [pm.callbacks.CheckParametersConvergence(diff="absolute")]
        }
    else:
        assert False, "Unknown convergence criterion."

    with m as model:

        fit_res = pm.fit(method=method, n=n_steps, **extra_args)
        draws = fit_res.sample(n_draws)

    return {"draw_dict": arviz_to_draw_dict(draws), "fit_res": fit_res}
