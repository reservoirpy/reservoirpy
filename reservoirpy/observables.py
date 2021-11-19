import numpy as np

from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs


def spectral_radius(W, maxiter: int = None) -> float:
    """Compute the spectral radius of a matrix `W`.

    Spectral radius is defined as the maximum absolute
    eigenvalue of `W`.

    Parameters
    ----------
    W : numpy.ndarray or scipy.sparse matrix
        Matrix from which the spectral radius will
        be computed.

    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        By default, equal to `W.shape[0] * 20`.
        See `Scipy documentation <https://docs.scipy.org/
        doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_
        for more informations.

    Returns
    -------
    float
        Spectral radius of `W`.

    Raises
    ------
    ArpackNoConvergence
        When computing spectral radius on large
        sparse matrices, it is possible that the
        Fortran ARPACK algorithmn used to compute
        eigenvalues don't converge towards precise
        values. To avoid this problem, set the `maxiter`
        parameter to an higher value. Be warned that
        this may drastically increase the computation
        time.

    """
    if issparse(W):
        if maxiter is None:
            maxiter = W.shape[0] * 20

        return max(abs(eigs(W,
                            k=1,
                            which='LM',
                            maxiter=maxiter,
                            return_eigenvectors=False)))

    return max(abs(linalg.eig(W)[0]))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def nrmse(y_true, y_pred, norm="minmax", norm_value=None):
    error = rmse(y_true, y_pred)
    if norm_value is not None:
        return error / norm_value

    else:
        norms = {"minmax": lambda y: y.ptp(),
                 "var" : lambda y: y.var(),
                 "mean": lambda y: y.mean(),
                 "q1q3": lambda y: np.quantile(y, 0.75) - np.quantile(y, 0.25)}

        if norms.get(norm) is None:
            raise ValueError(f"Unknown normalization method. "
                             f"Available methods are {list(norms.keys())}.")
        else:
            return error / norms[norm](y_true)


def rsquare(y_true, y_pred):
    d = (y_true - y_pred) ** 2
    D = (y_true - y_true.mean())**2
    return 1 - np.sum(d) / np.sum(D)
