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


def compute_error_NRMSE(teacher_signal, predicted_signal, verbose=False):
    """ Computes Normalized Root-Mean-Squarred Error between a teacher signal and a predicted signal
    Return the errors in this order: nmrse mean, nmrse max-min, rmse, mse.
    By default, only NMRSE mean should be considered as a general measure to be
    compared for different datasets.

    For more information, see:
    - Mean Squared Error https://en.wikipedia.org/wiki/Mean_squared_error
    - Root Mean Squared Error https://en.wikipedia.org/wiki/Root-mean-square_deviation for more info
    """
    errorLen = len(predicted_signal[:])
    mse = np.mean((teacher_signal - predicted_signal)**2)
    rmse = np.sqrt(mse)
    nmrse_mean = abs(rmse / np.mean(predicted_signal[:])) # Normalised RMSE (based on mean)
    nmrse_maxmin = rmse / abs(np.max(predicted_signal[:]) - np.min(predicted_signal[:])) # Normalised RMSE (based on max - min)
    if verbose:
        print("Errors computed over %d time steps" % (errorLen))
        print("\nMean Squared error (MSE):\t\t%.4e" % (mse) )
        print("Root Mean Squared error (RMSE):\t\t%.4e\n" % rmse )
        print("Normalized RMSE (based on mean):\t%.4e" % (nmrse_mean) )
        print("Normalized RMSE (based on max - min):\t%.4e" % (nmrse_maxmin) )
    return nmrse_mean, nmrse_maxmin, rmse, mse
