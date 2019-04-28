import numpy as np
import pandas as pd
from scsg import SCSG


def f_1(a, b):
    """
    Example function.
    :param x:
    :param y:
    :return:
    """
    return a**2 + b**2


def f_2(a, b):
    return a + b


def f_3(a, b):
    return a * b


f = np.array([f_1, f_2, f_3])
number_stages_T = 10
x_0 = (3, 5)
n = 3
list_of_x_hat = np.array([x_0])
list_of_x = np.array([])
stepsizes = np.random.randint(3, size=number_stages_T - 1)
batch_sizes = np.random.randint(low=1, high=n, size=number_stages_T - 1)
mini_batch_sizes = np.random.randint(low=1, high=n, size=number_stages_T - 1)

mySCSG = SCSG(number_stages_T,
              x_0,
              stepsizes,
              batch_sizes,
              mini_batch_sizes,
              f,
              n=3)

test = mySCSG.run()
