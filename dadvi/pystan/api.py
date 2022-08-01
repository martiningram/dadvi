import numpy as np
import pystan
from dadvi.objective_from_model import build_dadvi_funs
from dadvi.core import find_dadvi_optimum, get_dadvi_draws


def get_wrapped_log_prob(log_prob):
    
    def new_log_prob(params):
    
        try:
            return log_prob(params)
        except ValueError:
            return np.nan
        
    return new_log_prob
    

def get_wrapped_gradient(grad_log_prob):
    
    def new_grad_log_prob(params):    
        try:
            return grad_log_prob(params)
        except ValueError:
            return np.repeat(np.nan, params.shape[0])
        
    return new_grad_log_prob


def unflatten_draws_pystan(pystan_fit_res, flat_draws):

    # Transform the parameters into the full vectors
    constrained_pars = np.array([pystan_fit_res.constrain_pars(cur_draw) for
                                 cur_draw in flat_draws])

    rel_draws = constrained_pars.copy()
    param_draws = dict()

    for cur_par, cur_shape in zip(pystan_fit_res.model_pars,
                                  pystan_fit_res.par_dims):

        to_pick = int(np.prod(cur_shape))

        cur_rel_draws = rel_draws[:, :to_pick]

        # TODO: Is this the correct reshaping order? Pretty sure it's row-major,
        # i.e. Fortran / R style.
        reshaped = cur_rel_draws.reshape(-1, *cur_shape, order='F')

        rel_draws = rel_draws[:, to_pick:]

        param_draws[cur_par] = reshaped

    return param_draws


def fit_dadvi_pystan(model_code, model_data, draws=1000, seed=2,
                     n_fixed_draws=50, verbose=False):

    np.random.seed(seed)

    model = pystan.StanModel(model_code=model_code)

    # TODO: Make this quiet
    fit_res = model.sampling(data=model_data, warmup=1, iter=2)

    n_pars_total = len(fit_res.unconstrained_param_names())

    stan_dadvi_funs = build_dadvi_funs(get_wrapped_log_prob(fit_res.log_prob), 
                                       get_wrapped_gradient(fit_res.grad_log_prob))

    opt = find_dadvi_optimum(np.concatenate([np.zeros(n_pars_total),
                                             np.repeat(-3., n_pars_total)]), 
                             np.random.randn(n_fixed_draws, n_pars_total),
                             stan_dadvi_funs, opt_method='L-BFGS-B', verbose=verbose)

    opt_pars = opt['opt_result'].x

    draws = get_dadvi_draws(opt_pars, np.random.randn(1000, opt_pars.shape[0] // 2))

    param_dict = unflatten_draws_pystan(fit_res, draws)

    return param_dict, opt
