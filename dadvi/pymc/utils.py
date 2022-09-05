import aesara.tensor as at
import pymc as pm
import aesara


def get_hvp(m):
    # TODO: Will have to get the input arguments right

    b = at.vector(name="b")
    hessian = m.d2logp()
    vars = pm.aesaraf.cont_inputs(hessian)
    hvp = hessian @ b

    hvp_fn = aesara.function(vars + [b], [hvp])

    return hvp_fn


def get_value_and_grad(m):

    val_and_grad = m.logp_dlogp_function()
    val_and_grad.set_extra_values({})

    return val_and_grad


def arviz_to_draw_dict(az_trace):
    # Converts an arviz trace to a dict of str -> np.ndarray

    dict_version = dict(az_trace.posterior.data_vars.variables)

    return {x: y.values for x, y in dict_version.items()}
