import numpy as np
import uncertainties.unumpy as unp


def linregress(x, y):
    assert len(x) == len(y)


    N = len(y)
    Delta = N * sum(x**2) - (sum(x))**2

    A = (N * sum(x * y) - sum(x) * sum(y)) / Delta
    B = (sum(x**2) * sum(y) - sum(x) * sum(x * y)) / Delta

    sigma_y = unp.sqrt(sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * unp.sqrt(N / Delta)
    B_error = sigma_y * unp.sqrt(sum(x**2) / Delta)

    return [A, B], [A_error, B_error], sigma_y
