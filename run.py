import numpy as np 
from utils.helpers import load_csv_data
from implementations import *

x_tr, x_te, y_tr, tr_ids, te_ids = load_csv_data("data/")

N = x_tr.shape[0]
D = x_tr.shape[1]
initial_w = np.zeros(D + 1) # will add bias term to x, hence D + 1 dimension

# --- PRE-PROCESSING ---

# remove Nan values
col_medians = np.nanmedian(x_tr, axis=0)
col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
indices = np.where(np.isnan(x_tr))
x_tr[indices] = np.take(col_medians, indices[1])

# standardize the data
mean = np.mean(x_tr, axis=0)
std = np.std(x_tr, axis=0)
std[std == 0] = 1 
x_tr = (x_tr - mean) / std

# add bias term
x_tr = np.c_[np.ones(N), x_tr]

# set labels from 0 to 1 to re-use implementation of logistic regression from the course
y_tr_01 = np.where(y_tr == -1, 0, 1)

# --- TRAINING ---
w, loss = logistic_regression(y_tr_01, x_tr, initial_w, 5000, 0.01)
print(f"Final model : loss = {loss}")