import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import *

plt.style.use('ggplot')

__author__ = 'kqureshi'

a, b = 0, 1
n, beta, tmax, t_weight = 3, 2, 500, 1
w = 1 / n


def sig(x: float, beta: float = 2) -> float:
    return 1 / (1 + np.exp(beta * x))


class Gradients:
    def __init__(self):
        beta = 2

    def softmax(self, x: float) -> float:
        return np.exp(self.beta * x) / (np.exp(self.beta * x) + 1)

    def trust(self, x: float) -> float:
        return (-self.beta * (x / (1 - x)) ** self.beta) / ((x - 1) * x * (1 + (x / (1 - x)) ** self.beta) ** 2)

    def sigmoid(self, x: float) -> float:
        return (sig(x, self.beta)) * (1 - (sig(x, self.beta)))

    def tanh(self, x: float) -> float:
        return (4 * np.exp(-2 * x)) / (np.exp(-2 * x) + 1) ** 2

    def arctan(self, x: float) -> float:
        return 1 / (1 + x ** 2)


def sim(func, init: np.ndarray, tmax: int, n: int, stubborn: bool = False) -> pd.DataFrame:
    Theta = init
    for t in range(1, tmax):
        theta_old = Theta[t - 1, :]
        theta_new = theta_old.copy()
        p = int(np.ceil(np.random.rand(1) * (n - 1))[0])
        for i in range(n):
            if ((i != p) & ((theta_old[i] - theta_old[p]) != 0) & (np.abs(theta_old[i] - theta_old[p]) != 1) & (
                        theta_old[i] != 1) & (theta_old[i] != 0)):
                if theta_old[p] > theta_old[i]:
                    theta_new[i] = theta_old[i] + (
                        w * t_weight * (func(theta_old[i] - theta_old[p])) * (func(theta_old[i])))
                else:
                    theta_new[i] = theta_old[i] - (
                        w * t_weight * (func(theta_old[i] - theta_old[p])) * (func(theta_old[i])))
            else:
                theta_new[i] = theta_old[i]
        if stubborn:
            theta_new[n - 1] = max(0, theta_new[n - 2] - 0.5)  # Stubborn agent
        for j in range(n):
            if theta_new[j] > 1:
                theta_new[j] = 1
            elif theta_new[j] < 0:
                theta_new[j] = 0
        Theta[t] = theta_new
    return pd.DataFrame(Theta, columns=['agent_' + str(j) for j in range(n)])


def plot(data: pd.DataFrame, function: str, n: int, beta: float, time: int) -> None:
    pd.DataFrame(data, columns=['agent_' + str(j) for j in range(n)]).plot()
    plt.ylabel('$\Theta_i(t)$')
    plt.xlabel('$t$')
    plt.savefig(os.environ['HOME'] + '/Dropbox (MIT)/optimal_control/figures/{}_{}n_{}b_{}t.png'.format(
        function, str(n), str(beta), str(time)))


class DoubleGradient:
    def __init__(self):
        beta = 2

    def trust(self, x: float):
        return (self.beta * (x / (1 - x)) ** self.beta * ((x / (1 - x)) ** self.beta * (2 * x - self.beta - 1
                                                                                        ) + 2 * x + self.beta - 1)) / (
                   (x - 1) ** 2 * x ** 2 * (1 + (x / (1 - x)) ** self.beta) ** 3)


def main(function: str = 'trust') -> None:
    theta = np.zeros([tmax, n])
    theta[0] = (a + ((b - a) * (np.random.rand(n, 1)))).T
    plot(data=sim(func=Gradients().softmax, init=theta), function=function, n=n, beta=beta, time=tmax)
