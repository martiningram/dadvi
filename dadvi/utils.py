from scipy.sparse.linalg import cg, LinearOperator


def cg_using_fun_scipy(A_dot_x, b, preconditioner):

    n_params = b.shape[0]

    # Wrap with LinearOperator
    op = LinearOperator((n_params, n_params), matvec=A_dot_x)

    return cg(op, b, M=preconditioner)
