import numpy as np
from utils.helpers import batch_iter
from utils.costs import compute_mse
from utils.costs import neg_log_likelihood
from utils.gradient import compute_gradient
from utils.gradient import gradient_neg_ll

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last computed parameter
        loss: the loss associated to w
    """
    w = initial_w.copy()

    # update w
    for _ in range(max_iters):
        grad, _ = compute_gradient(y, tx, w)
        w = w - gamma * grad
    
    # compute loss for last w
    loss = compute_mse(y, tx, w)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD). 
        Note that we use batch size = 1 for this version of SGD.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last computed parameter
        loss: the loss associated to w
    """

    # Define parameters 
    batch_size = 1
    w = initial_w.copy()

    # update w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            stoch_grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * stoch_grad

    loss = compute_mse(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
        returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N, ), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D, ), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, w)
    return w, mse
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N, ), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D, ), D is the number of features.
        loss: the cost of the fitting, usually mse.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212]), 5.585196838722984e-32
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628]), 0.006417022643777357
    """
    N, D = tx.shape
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.identity(D), tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implements logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N, ), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D, ), the initial weights of the model.
        max_iters: scalar, the maximum number of iterations of the model.
        gamma: scalar, the step-size used for gradient descent.

    Returns:
        w: optimal weights, numpy array of shape(D, ), D is the number of features.
        loss: the cost of the associated weights, here the cost function is the negative log likelihood.
    """
    threshold = 1e-8
    w = initial_w.copy()
    previous_loss = neg_log_likelihood(y, tx, w)
    loss = previous_loss

    for _ in range(max_iters):
        # update w
        gradient = gradient_neg_ll(y, tx, w)
        w = w - gamma * gradient
        loss = neg_log_likelihood(y, tx, w)
        # if the update is not significant enough, stop iterating
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implements regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N, ), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, the regularizer parameter.
        initial_w: numpy array of shape (D, ), the initial weights of the model.
        max_iters: scalar, the maximum number of iterations of the model.
        gamma: scalar, the step-size used for gradient descent.

    Returns:
        w: optimal weights, numpy array of shape(D, ), D is the number of features.
        loss: the cost of the associated weights, here the cost function is the negative log likelihood.
    """
    threshold = 1e-8
    w = initial_w.copy()
    previous_loss = neg_log_likelihood(y, tx, w)
    loss = previous_loss

    for _ in range(max_iters):
        # update w : gradient regularized, loss not
        gradient = gradient_neg_ll(y, tx, w) + lambda_ * 2 * w
        w = w - gamma * gradient
        loss = neg_log_likelihood(y, tx, w)
        # if the update is not significant enough, stop iterating
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss
    
    return w, loss