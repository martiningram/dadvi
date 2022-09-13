import sys
import time
from os.path import join
from os import makedirs

import pymc as pm
import numpy as np
import pandas as pd
import pickle

from utils import load_model_by_name, arviz_to_draw_dict, fit_pymc_sadvi

if __name__ == "__main__":

    model_name = sys.argv[1]
    method = sys.argv[2]
    model = load_model_by_name(model_name)

    print("Fitting")
    start_time = time.time()

    with model as m:
        fit_result_sadvi = fit_pymc_sadvi(
            m, n_draws=1000, n_steps=100000, method=method, convergence_crit="default"
        )

    end_time = time.time()
    runtime = end_time - start_time
    print("Done")

    base_dir = f"s{method}_results"

    makedirs(join(base_dir, "draw_dicts"), exist_ok=True)
    makedirs(join(base_dir, "info"), exist_ok=True)

    np.savez(
        join(base_dir, "draw_dicts", model_name + ".npz"),
        **fit_result_sadvi["draw_dict"],
    )

    with open(join(base_dir, "info", model_name + ".pkl"), "wb") as f:
        pickle.dump(
            {
                "steps": fit_result_sadvi["fit_res"].hist.shape[0],
                "runtime": runtime,
            },
            f,
        )
