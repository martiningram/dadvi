from typing import NamedTuple, Callable, Optional
from .optimization import optimize_with_hvp
from functools import partial


class DADVIFuns(NamedTuple):

    # Function of eta (variational parameters) and zs (draws)
    # zs should have shape (M, D), where M is number of fixed draws and D is
    # problem dimension.
    kl_est_and_grad_fun: Callable

    # Function of eta, zs, and b, a vector to compute the hvp with
    kl_est_hvp_fun: Optional[Callable]


def find_dadvi_optimum(
    init_params, zs, dadvi_funs, opt_method="trust-ncg", callback_fun=None
):

    val_and_grad_fun = lambda var_params: dadvi_funs.kl_est_and_grad_fun(var_params, zs)
    hvp_fun = (
        None
        if dadvi_funs.kl_est_hvp_fun is None
        else lambda var_params, b: dadvi_funs.kl_est_hvp_fun(var_params, zs, b)
    )

    opt_result, eval_count = optimize_with_hvp(
        val_and_grad_fun,
        hvp_fun,
        init_params,
        opt_method=opt_method,
        callback_fun=callback_fun,
    )

    return {"opt_result": opt_result, "evaluation_count": eval_count}
