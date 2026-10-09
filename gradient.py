# -*- coding: utf-8 -*-

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w, 
        and the error vector.
    """

    e = y - tx @ w
    grad = - (tx.T @ e) / e.shape[0]
    return grad, e
