"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

import numpy.linalg


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    y = mixture
    probs = np.zeros((X.shape[0], len(y[1])))

    for i in range(X.shape[0]):

        for j in range(len(y[1])):

            C = y[2][j] / ((2*np.pi*y[1][j])**(X.shape[1]/2))

            probs[i, j] = C * \
                np.exp(-numpy.linalg.norm(X[i]-y[0][j])**2/(2*y[1][j]))

    xsums = np.sum(probs, axis=1)

    return (1/xsums)[:, None]*probs, np.sum(np.log(xsums))


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    nums = np.sum(post, axis=0)

    probs = nums / X.shape[0]

    mus = (1/nums)[:, None] * np.dot(X.T, post).T

    var = np.zeros(post.shape[1])

    for j in range(post.shape[1]):
        var[j] = (1/(nums[j]*X.shape[1])) * \
            np.dot((numpy.linalg.norm(X-mus[j], axis=1)**2), post[:, j]).T

    return GaussianMixture(mus, var, probs)


def run_em(X: np.ndarray, mixture: GaussianMixture,
           post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:s
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
    mixture = mstep(X, probs)

    while True:

        probs, newloglike = estep(X, mixture)
        mixture = mstep(X, probs)

        if (newloglike-loglike) <= (10**(-6)*np.abs(newloglike)):
            break

        else:
            loglike = newloglike

    return mixture, probs, newloglike
