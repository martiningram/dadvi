import pandas as pd
from load_data import load_arm_model, add_bambi_family
import sys
from os import makedirs
from os.path import join
import numpy as np


def arviz_to_draw_dict(az_trace):
    # Converts an arviz trace to a dict of str -> np.ndarray

    dict_version = dict(az_trace.posterior.data_vars.variables)

    return {x: y.values for x, y in dict_version.items()}


if __name__ == "__main__":

    df = pd.read_csv(
        "/Users/martin.ingram/Projects/PhD/SharedExampleModels/ARM/rstanarm_ij_configs.csv"
    )

    df = add_bambi_family(df)

    model_name = sys.argv[1]

    # Should check for duplicates here
    rel_row = df[df["model_name"] == model_name].iloc[0]
    model = load_arm_model(
        rel_row, "/Users/martin.ingram/Projects/PhD/SharedExampleModels/ARM_json/"
    )["bambi_model"]

    print(model)
    print("Fitting")
    fit_result_nuts = model.fit()
    print("Done")

    makedirs("nuts_results/netcdfs", exist_ok=True)
    makedirs("nuts_results/draw_dicts", exist_ok=True)

    fit_result_nuts.to_netcdf(join("nuts_results", "netcdfs", model_name + ".netcdf"))
    draw_dict = arviz_to_draw_dict(fit_result_nuts)
    np.savez(join("nuts_results/draw_dicts", model_name + ".npz"), **draw_dict)
