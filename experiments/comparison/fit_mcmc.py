import sys
from os import makedirs
from os.path import join
import numpy as np
import time
import pandas as pd
import pymc as pm


def arviz_to_draw_dict(az_trace):
    # Converts an arviz trace to a dict of str -> np.ndarray

    dict_version = dict(az_trace.posterior.data_vars.variables)

    return {x: y.values for x, y in dict_version.items()}


if __name__ == "__main__":

    from utils import load_model_by_name

    model_name = sys.argv[1]
    model = load_model_by_name(model_name)

    print("Fitting")
    start_time = time.time()

    with model as m:
        fit_result_nuts = pm.sample()

    end_time = time.time()
    runtime = end_time - start_time
    print("Done")

    makedirs("nuts_results/netcdfs", exist_ok=True)
    makedirs("nuts_results/runtimes", exist_ok=True)
    makedirs("nuts_results/draw_dicts", exist_ok=True)

    fit_result_nuts.to_netcdf(join("nuts_results", "netcdfs", model_name + ".netcdf"))
    draw_dict = arviz_to_draw_dict(fit_result_nuts)
    np.savez(join("nuts_results/draw_dicts", model_name + ".npz"), **draw_dict)

    pd.Series({"runtime": runtime}).to_csv(
        join("nuts_results", "runtimes", model_name + ".csv")
    )
