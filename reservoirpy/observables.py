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


def mse(y: np.ndarray, ypred: np.ndarray) -> float:
    """[summary]

    Parameters
    ----------
    y : np.ndarray
        [description]
    ypred : np.ndarray
        [description]

    Returns
    -------
    float
        [description]
    """
    return np.mean((y - ypred)**2)


def rmse(y: np.ndarray, ypred: np.ndarray) -> float:
    """[summary]

    Parameters
    ----------
    y : np.ndarray
        [description]
    ypred : np.ndarray
        [description]

    Returns
    -------
    float
        [description]
    """
    return np.sqrt(mse(y, ypred))


def nrmse(y: np.ndarray,
          ypred: np.ndarray,
          method: str = 'minmax',
          feat_axis: int = 1) -> float:
    """[summary]

    Parameters
    ----------
    y : np.ndarray
        [description]
    ypred : np.ndarray
        [description]
    method : str, optional
        [description], by default 'minmax'
    feat_axis : int, optional
        [description], by default 1

    Returns
    -------
    float
        [description]
    """

    if method == 'dev':
        ymean = np.mean(y, axis=feat_axis).reshape(-1, 1)
        numerator = np.mean((y - ypred)**2)
        denominator = np.mean((y - ymean)**2)
        return np.sqrt(numerator / denominator)

    err = rmse(y, ypred)

    if method == 'minmax':
        return err / (np.max(y) - np.min(y))
    if method == 'mean':
        return err / np.mean(y)


def r2_coeff(y: np.ndarray, ypred: np.ndarray) -> float:
    """[summary]

    Parameters
    ----------
    y : np.ndarray
        [description]
    ypred : np.ndarray
        [description]

    Returns
    -------
    float
        [description]
    """
    ymean = np.mean(y)
    numerator = np.sum((y - ypred)**2)
    denominator = np.sum((y - ymean)**2)
    return 1.0 - numerator/denominator
