"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import numpy.linalg


def safe_ln(x):
    if x <= 0:
        return 0
    return np.log(x)


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    mask = (X == 0)
    X_m = np.ma.array(X, mask=mask)

    x_i = X_m[:, None, :]
    y_j = mixture.mu[None, :, :]
    int_sum = (x_i - y_j) ** 2
    sigma = mixture.var * np.eye(len(mixture.var))
    invsigma = np.linalg.inv(sigma)

    probs = np.zeros((X.shape[0], mixture.mu.shape[0]))
    xsums = np.zeros(X.shape[0])

    onem = np.ones((X.shape[1], mixture.var.shape[0]))

    for i in range(X.shape[0]):

        under_exp = -0.5 * np.dot(invsigma, int_sum[i])
        smat = mixture.var * (onem)
        coef_exp = np.sqrt(1/(2*np.pi*smat.T))
        log_coef = np.log(coef_exp)*(~int_sum[i].mask)
        logp = np.log(mixture.p+1e-16)
        probs[i] = logp + np.sum(under_exp+log_coef, axis=1)
        xsums[i] = logsumexp(probs[i])

    logsums = np.sum(xsums)

    return np.exp(probs-xsums[None, :].T), logsums


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    nums = np.sum(post, axis=0)
    probs = nums / X.shape[0]

    mask = (X == 0)
    X_m = np.ma.array(X, mask=mask)

    munorm = np.dot(~X_m.mask.T, post).T
    maskmu = (munorm >= 1)

    mus = ((np.dot(X_m.T, post).T)/munorm) * maskmu + (mixture.mu) * (~maskmu)

    x_i = X_m[:, None, :]
    y_j_n = mus[None, :, :]

    D_ij_n = np.sum(((x_i - y_j_n)**2), axis=2)

    var = (1/np.dot(post.T, np.sum(~X_m.mask, axis=1))) *\
        np.sum((D_ij_n*post), axis=0).data

    var = (var < min_variance) * min_variance *\
        np.ones(var.shape) + (~(var < min_variance)) * var

    return GaussianMixture(mus.data, var, probs)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    probs, loglike = estep(X, mixture)

    mixture = mstep(X, probs, mixture)

    while True:

        probs, newloglike = estep(X, mixture)
        mixture = mstep(X, probs, mixture)

        if newloglike-loglike <= 10**(-6)*np.abs(newloglike):
            break

        else:
            loglike = newloglike

    return mixture, probs, newloglike


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    probs = estep(X, mixture)[0]

    pred = np.zeros_like(X)
    for i in range(X.shape[0]):
        pred[i] = np.dot(mixture.mu.T, probs[i])

    mask = (X != 0)

    return pred * (~mask) + X * (mask)
