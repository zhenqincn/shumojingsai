import numpy as np
import math


def SNR(ps, pn):
    return ps - pn


def loss(distance=80):
    return distance * 0.2


def gain(distance=80):
    return distance * 0.2


def Pam(length):
    return 2 * math.pi * 193.1 * 1e12 * 6.62606896 * 1e-34 * 50 * 1e9 * (4 - 1 / (math.pow(10, (gain(length) / 10))))


def Ps(n):
    if n == 0:
        return 10        # 单位为dBm
    else:
        return Ps(n - 1) - loss() + gain()


def Pn(n):
    if n == 0:
        return -20
    else:
        return Pn(n - 1) + Pnl(n - 1, K(80)) + Pam(80) - loss() + gain()


def Pnl(n, coefficient):
    return coefficient * math.pow((Ps(n) + Pn(n)), 2)


def K(length):
    if length == 80:
        return 0.10651750493008656
    elif length == 100:
        return 0.10692265508407804


