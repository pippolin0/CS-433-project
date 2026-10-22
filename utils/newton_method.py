import numpy as np
from helpers import sigmoid
from costs import neg_log_likelihood
from gradient import gradient_neg_ll

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y, tx, w)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """
    pred = sigmoid(tx @ w)
    weights = pred * (1 - pred)
    S = np.diagflat(weights)
    return 1 / y.shape[0] * tx.T @ S @ tx

    # computed without explicitly constructing S
    #return (tx.T @ (weights * tx)) / y.shape[0]

def newton_method(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    w = initial_w.copy()
    previous_loss = neg_log_likelihood(y, tx, w)
    loss = previous_loss

    for _ in range(max_iters):
        # update w
        gradient = gradient_neg_ll(y, tx, w)
        hessian = calculate_hessian(y, tx, w)
        w = np.linalg.solve(hessian, hessian @ w - gamma * gradient)
        loss = neg_log_likelihood(y, tx, w)
        # if the update is not significant enough, stop iterating
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss
    
    return w, loss
