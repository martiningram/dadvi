import pymc3 as pm
import numpy as np


def get_basic_2d_mvn():

    corr = np.array([[1., 0.9],
                    [0.9, 1.]])

    sds = np.array([2, 1]).reshape(-1, 1)

    cov = sds.T * corr * sds

    with pm.Model() as m:
    
        x = pm.MvNormal('x', mu=[2, 2], cov=cov, shape=2)

    return m
