import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from utils.helpers import load_csv_data, create_csv_submission, sigmoid
from logistic_regression.logistic_cross_validation import cross_validate_reg_logistic_regression
from utils.metrics import compute_f1, compute_accuracy
from implementations import reg_logistic_regression

# --- LOAD DATA ---

print("Loading dataset...")
x_tr, x_te, y_tr, tr_ids, te_ids = load_csv_data(os.path.join(PROJECT_ROOT, "data"), sub_sample=False)
print(f"Raw shapes: x_tr={x_tr.shape}, y_tr={y_tr.shape}")

# --- PRE-PROCESSING ---

# Remove NaN values using training column medians for both train and test
col_medians_tr = np.nanmedian(x_tr, axis=0)
col_medians_tr = np.where(np.isnan(col_medians_tr), 0.0, col_medians_tr)
indices_tr = np.where(np.isnan(x_tr))
x_tr[indices_tr] = np.take(col_medians_tr, indices_tr[1])

indices_te = np.where(np.isnan(x_te))
x_te[indices_te] = np.take(col_medians_tr, indices_te[1])

# Standardize the data using training mean and std for both train and test
mean_tr = np.mean(x_tr, axis=0)
std_tr = np.std(x_tr, axis=0)
std_tr[std_tr == 0] = 1
x_tr = (x_tr - mean_tr) / std_tr
x_te = (x_te - mean_tr) / std_tr

# Add bias term
x_tr = np.c_[np.ones(x_tr.shape[0]), x_tr]
x_te = np.c_[np.ones(x_te.shape[0]), x_te]

# --- CROSS-VALIDATION ---

#gammas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
gammas = [0.05, 0.1, 0.15]
#gammas = [0.2, 0.4, 0.5, 0.6]
#gammas = [0.6]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2]
taus = np.linspace(0.15, 0.50, 100)

cv_results = cross_validate_reg_logistic_regression(
    y=y_tr,
    tx=x_tr,
    gammas=gammas,
    lambdas=lambdas,
    taus=taus,
    k_fold=5,
    max_iters=500,
    use_full_preprocessor=True,
)

print(
    f"\nBest CV hyperparameters:\n"
    f"  - gamma*  = {cv_results['best_gamma']}\n"
    f"  - lambda* = {cv_results['best_lambda']}\n"
    f"  - tau*    = {cv_results['best_tau']:.2f}\n"
    f"  - CV F1   = {cv_results['best_f1']:.4f}"
)

# --- FINAL TRAINING & SUBMISSION ---

y_tr_01 = np.where(y_tr == -1, 0.0, 1.0)
initial_w = np.zeros(x_tr.shape[1])

w, final_loss = reg_logistic_regression(
    y_tr_01,
    x_tr,
    cv_results["best_lambda"],
    initial_w,
    max_iters=2000,
    gamma=cv_results["best_gamma"],
)

probs_tr = sigmoid(x_tr @ w)
y_pred_tr = np.where(probs_tr >= cv_results["best_tau"], 1, -1)
print(
    f"Final model on training set: loss={final_loss:.4f}, "
    f"F1={compute_f1(y_tr, y_pred_tr):.4f}, Acc={compute_accuracy(y_tr, y_pred_tr):.4f}"
)

probs_te = sigmoid(x_te @ w)
pred_te = np.where(probs_te >= cv_results["best_tau"], 1, -1)

create_csv_submission(te_ids, pred_te, "reg_logistic-with-cv-full-pp.csv")
