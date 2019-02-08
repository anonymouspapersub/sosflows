import numpy as np
import scipy.stats as ss
import torch.distributions
from torch.distributions import Normal, Categorical


class mixNormal:
    def __init__(self, num_components):
        self.mean = np.random.randint(-10, 10, num_components) + np.random.normal(0, 3, num_components)
        self.scale = np.abs(np.random.randint(0, 3, num_components) + np.random.normal(0, 1, num_components))
        self.weights = np.random.randint(1,10, num_components)
        self.weights = self.weights/ np.sum(self.weights)
        self.components = num_components


class mixNormalFixed:
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale
        self.weights = np.ones(np.size(self.mean))/np.size(self.mean)
        self.components = np.size(self.mean)


def bisection(xmin, xmax, F, tol, mean, scale, weights, n_components):
    x = (xmin + xmax) / 2.0
    Fx = 0
    for i in range(0, n_components):
        Fx += weights[i] * ss.norm.cdf(x, loc=mean[i], scale=scale[i])
    while np.abs(Fx - F) > tol:
        if Fx == F:
            return x
        elif Fx < F:
            xmin = x
        else:
            xmax = x
        x = (xmin + xmax) / 2.0
        Fx = 0
        for i in range(0, n_components):
            Fx += weights[i] * ss.norm.cdf(x, loc=mean[i], scale=scale[i])
    return x


def transform_bisec(source, target, n):
    Fz, Fx, fz, fx = np.zeros(4)
    z = np.linspace(-10, 10, num=n)
    for i in range(0, source.components):
        Fz += source.weights[i] * ss.norm.cdf(z, loc=source.mean[i], scale=source.scale[i])
        fz += source.weights[i] * ss.norm.pdf(z, loc=source.mean[i], scale=source.scale[i])

    x = np.linspace(-10, 10, (np.size(Fz, 0)))
    for i in range(0, target.components):
        Fx += target.weights[i] * ss.norm.cdf(x, loc=target.mean[i], scale=target.scale[i])
        fx += target.weights[i] * ss.norm.pdf(x, loc=target.mean[i], scale=target.scale[i])
    '''
    EVALUATE TRANSFORAMTION 
    '''
    X = np.zeros(np.size(Fz))
    for i in range(np.size(Fz)):
        X[i] = bisection(-30, 30, Fz[i], 10 ** -10, target.mean, target.scale, target.weights,
                                             target.components)
    '''
    SLOPE OF TRANSFORMATION 
    '''
    qu = 0
    for i in range(0, target.components):
        qu += target.weights[i] * ss.norm.pdf(X, loc=target.mean[i], scale=target.scale[i])
    gradT = np.divide(fz, qu)
    return z, fz, x, fx, X, gradT


def standard2mix(target, mixture, standard):
    n_components = np.size(target.scale)
    x = np.linspace(-15, 10, num=100000)
    y = np.linspace(-10, 10, num=50000)
    Fx = 0
    fx = 0
    if mixture == 'gaussian':
        for i in range(0, n_components):
            Fx += target.weights[i]*ss.norm.cdf(x, loc=target.mean[i], scale=target.scale[i])
            fx += target.weights[i]*ss.norm.pdf(x,loc=target.mean[i],scale=target.scale[i])
    elif mixture == 'laplacian':
        for i in range(0, n_components):
            Fx += target.weights[i] * ss.laplace.cdf(x, loc=target.mean[i], scale=target.scale[i])
            fx += target.weights[i] * ss.laplace.pdf(x, loc=target.mean[i], scale=target.scale[i])
    else:
        df = np.abs(np.random.randint(1, 5, n_components) + np.random.normal(0, 1, n_components))
        for i in range(0, n_components):
            Fx += target.weights[i]*ss.t.cdf(x, df = df[i], loc=target.mean[i], scale=target.scale[i])
            fx += target.weights[i]*ss.t.pdf(x,df=df[i], loc=target.mean[i], scale=target.scale[i])

    if standard == 'gaussian':
        z = ss.norm.ppf(Fx, loc=0, scale=1)
        fz = ss.norm.pdf(y, loc=0, scale=1)
    elif standard == 'laplacian':
        z = ss.laplace.ppf(Fx, loc=0, scale=1)
        fz = ss.laplace.pdf(x, loc=0, scale=1)
    else:
        df = np.abs(np.random.randint(1, 5) + np.random.normal(0, 1))
        z = ss.t.ppf(Fx, df=1, loc=0, scale=1)
        fz = ss.t.pdf(x, df=1, loc=0, scale=1)

    return z, fz, x, fx, y


def make_clusters(params):
    return [Normal(*theta) for theta in params]


def sample_mixture(mixing_distribution, clusters, N):
    c = mixing_distribution.sample(torch.Size([N]))
    out = torch.zeros(N)
    for i in range(N):
        out[i] = clusters[c[i].item()].sample()
    return out