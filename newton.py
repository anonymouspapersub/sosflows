import torch
import torch.nn as nn

import math

pi = math.pi

def newton(x0, f, df, tolerance=1e-6):
    x = x0
    delta = torch.abs(f(x))
    while delta.max() > tolerance:
        x = x - f(x)/df(x)
        delta = torch.abs(f(x))
    return x

def newton_inverse(x, z0, f, df, tolerance=1e-6):
    g = lambda z: x - f(z)
    dg = lambda z: -df(z)
    return newton(z0, g, dg, tolerance)


class TestModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dim = d
        self.alpha = nn.Parameter(torch.randn(1,d))

    def forward(self, z):
        return self.alpha.dot(z**2)


model = TestModel(8)






