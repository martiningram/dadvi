import sys
from os import makedirs
from os.path import join
import numpy as np
import time
import pandas as pd

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.set_start_method("fork")

    import pymc as pm
    from utils import load_model_by_name, arviz_to_draw_dict

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
