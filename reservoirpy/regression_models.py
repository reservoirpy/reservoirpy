import numpy as np
import sklearn.linear_model as sklm

from scipy import linalg


def sklearn_linear_model(model):

    def linear_model_solving(X, Y):
            """
                Uses regression method provided during network instanciation to return W such as W * X ~= Ytarget
                First row of X MUST be only ones.
            """
            # Learning of the model (first row of X, which contains only ones, is removed)
            model.fit(X[1:, :].T, Y.T)

            # linear_model provides Matrix A and Vector b such as A * X[1:, :] + b ~= Ytarget
            A = np.asmatrix(model.coef_)
            b = np.asmatrix(model.intercept_).T

            # Then the matrix W = "[b | A]" statisfies "W * X ~= Ytarget"
            return np.asarray(np.hstack([b, A])) 
    
    return linear_model_solving


def ridge_linear_model(ridge, typefloat=np.float32):
    
    def ridge_model_solving(X, Y):
        ridgeid = (ridge*np.eye(X.shape[0])).astype(typefloat)
        
        return np.dot(np.dot(Y, X.T), linalg.inv(np.dot(X, X.T) + ridgeid))
    
    return ridge_model_solving


def pseudo_inverse_linear_model():
    
    def pseudo_inverse_model_solving(X, Y):
        return np.dot(Y, linalg.pinv(X))
    
    return pseudo_inverse_model_solving