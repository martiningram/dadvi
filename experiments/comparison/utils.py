import numpy as np
import pandas as pd
from config import ARM_CONFIG_CSV_PATH, ARM_JSON_PATH, MICROCREDIT_JSON_PATH, OCC_DET_PICKLE_PATH, POTUS_JSON_PATH
from dadvi.pymc.models.arm import load_arm_model, add_bambi_family
from dadvi.pymc.models.microcredit import load_microcredit_model
from dadvi.pymc.models.occ_det import get_occ_det_model_from_pickle
from dadvi.pymc.models.potus import get_potus_model


NON_ARM_MODELS = ["microcredit", "occ_det", "potus", "tennis"]


def load_model_by_name(model_name):

    if model_name == "microcredit":

        model = load_microcredit_model(MICROCREDIT_JSON_PATH)

    elif model_name == 'occ_det':

        model = get_occ_det_model_from_pickle(OCC_DET_PICKLE_PATH)

    elif model_name == 'potus':

        model = get_potus_model(POTUS_JSON_PATH)

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
