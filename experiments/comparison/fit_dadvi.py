# Fit using DADVI. This is the verbose version; we'll want a higher-level API down the road.
# It's not hard to write one, but hopefully this makes sense to you.
import sys
from utils import load_model_by_name, estimate_kl_fresh_draws
from dadvi.core import find_dadvi_optimum
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
import numpy as np
import time
from dadvi.utils import opt_callback_fun
from dadvi.core import get_dadvi_draws
from dadvi.pymc.pymc_to_jax import transform_dadvi_draws
from os import makedirs
from os.path import join
import pickle


model_name = sys.argv[1]
m = load_model_by_name(model_name)

# This will store the sequence of parameters
opt_callback_fun.opt_sequence = []

M = 30
seed = 2
np.random.seed(seed)

start_time = time.time()
jax_funs = get_jax_functions_from_pymc(m)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
init_means = np.zeros(jax_funs["n_params"])
init_log_vars = np.zeros(jax_funs["n_params"]) - 3
init_var_params = np.concatenate([init_means, init_log_vars])
zs = np.random.randn(M, jax_funs["n_params"])
opt = find_dadvi_optimum(
    init_params=init_var_params,
    zs=zs,
    dadvi_funs=dadvi_funs,
    verbose=True,
    callback_fun=opt_callback_fun,
)
finish_time = time.time()

runtime_dadvi = finish_time - start_time
dadvi_opt_sequence = opt_callback_fun.opt_sequence

z = np.random.randn(1000, jax_funs["n_params"])

dadvi_res = opt["opt_result"].x
dadvi_draws_flat = get_dadvi_draws(dadvi_res, z)

dadvi_dict = transform_dadvi_draws(
    m,
    dadvi_draws_flat,
    jax_funs["unflatten_fun"],
    add_chain_dim=True,
    keep_untransformed=True,
)

makedirs("dadvi_results/draw_dicts", exist_ok=True)
makedirs("dadvi_results/dadvi_info", exist_ok=True)
np.savez(join("dadvi_results/draw_dicts", model_name + ".npz"), **dadvi_dict)

kl_hist_dadvi = [
    estimate_kl_fresh_draws(dadvi_funs, cur_hist["theta"])
    for cur_hist in dadvi_opt_sequence
]

with open(join("dadvi_results", "dadvi_info", model_name + ".pkl"), "wb") as f:
    pickle.dump(
        {
            "opt_result": opt,
            "fixed_draws": zs,
            "M": zs.shape[0],
            "kl_hist": kl_hist_dadvi,
            "opt_sequence": dadvi_opt_sequence,
            "runtime": runtime_dadvi,
        },
        f,
    )
