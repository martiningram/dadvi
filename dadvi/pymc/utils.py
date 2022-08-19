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
