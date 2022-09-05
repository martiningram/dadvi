from scipy.sparse.linalg import cg, LinearOperator
from time import time


def cg_using_fun_scipy(A_dot_x, b, preconditioner):

    n_params = b.shape[0]

    # Wrap with LinearOperator
    op = LinearOperator((n_params, n_params), matvec=A_dot_x)

    return cg(op, b, M=preconditioner)


def opt_callback_fun(theta, val_and_grad_fun, hvp_fun):
    # Callback for jax_advi. It records only what will be needed
    # to estimate the KL later on.

    opt_callback_fun.opt_sequence.append(
        {
            "val_and_grad_calls": val_and_grad_fun.calls,
            "hvp_calls": hvp_fun.calls,
            "theta": theta,
            "time": time(),
        }
    )


opt_callback_fun.opt_sequence = list()
