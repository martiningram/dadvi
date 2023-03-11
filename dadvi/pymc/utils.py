import pytensor.tensor as pt
import pymc as pm
import pytensor


def get_hvp(m):
    # TODO: Will have to get the input arguments right

    b = pt.vector(name="b")
    hessian = m.d2logp()
    vars = pm.pytensorf.cont_inputs(hessian)
    hvp = hessian @ b

    hvp_fn = pytensor.function(vars + [b], [hvp])

    return hvp_fn


def get_value_and_grad(m):

    val_and_grad = m.logp_dlogp_function()
    val_and_grad.set_extra_values({})

    return val_and_grad


def arviz_to_draw_dict(az_trace):
    # Converts an arviz trace to a dict of str -> np.ndarray

    dict_version = dict(az_trace.posterior.data_vars.variables)

    return {x: y.values for x, y in dict_version.items()}


def get_unconstrained_variable_names(pymc_model):
    # Gets the names of the unconstrained parameters in the model (i.e. only the
    # ones required to compute the log density).

    rv_names = [rv.name for rv in pymc_model.value_vars]

    return rv_names
