import numpy as np 
from implementations import ridge_regression

# example used for adapting documentation of function
y = np.array([0.1,0.2])
tx = np.array([[2.3, 3.2], [1., 0.1]])
lambda_ = 1
print(ridge_regression(y, tx, lambda_))
