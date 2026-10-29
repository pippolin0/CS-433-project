# -*- coding: utf-8 -*-
"""K-Fold Cross-Validation utilities for hyperparameter (gamma, lambda_) and threshold (tau) tuning."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from implementations import logistic_regression, reg_logistic_regression
from utils.helpers import sigmoid
from utils.metrics import compute_f1, compute_accuracy
from logistic_regression.logistic_preprocessing import fit_preprocessor, transform_preprocessor


def build_k_indices(y, k_fold, seed=42):
    """Build k-fold index splits for cross-validation.

    Args:
        y: 1D numpy array of shape (N,)
        k_fold: int, number of folds K
        seed: int, random seed for reproducibility

    Returns:
        2D numpy array of shape (k_fold, N // k_fold) containing validation indices per fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validate_logistic_regression(
    y,
    tx,
    gammas,
    taus=None,
    k_fold=5,
    max_iters=500,
    seed=42,
    use_full_preprocessor=False,
    preprocess_kwargs=None,
):
    """Jointly cross-validate step-size (gamma) and decision threshold (tau) for logistic regression.

    Key insight:
    - Training `logistic_regression` only depends on `gamma` (NOT on `tau`), and `sigmoid` is unchanged!
    - For each fold and each `gamma`, we train `w` ONCE, compute `probs_val = sigmoid(tx_val @ w)` ONCE,
      and then evaluate all candidate thresholds `tau` (`y_pred = np.where(probs_val >= tau, 1, -1)`)
      in a few milliseconds without retraining.
    - When `use_full_preprocessor=True`, preprocessing (`fit_preprocessor` / `transform_preprocessor`)
      is fitted strictly inside each fold on the training split only (zero data leakage), and cached
      across `gammas` so each fold is only preprocessed once.

    Args:
        y: 1D numpy array of labels in {-1, 1}, shape (N,)
        tx: 2D numpy array of features, shape (N, D). Can be already pre-processed
            (when use_full_preprocessor=False) or raw (when use_full_preprocessor=True).
        gammas: list/1D array of candidate step sizes (gamma)
        taus: list/1D array of candidate decision thresholds in (0, 1). Defaults to [0.5] if None.
        k_fold: int, number of CV folds
        max_iters: int, number of gradient descent iterations
        seed: int, random seed for fold splitting
        use_full_preprocessor: bool. If False (default), uses `tx` directly as already preprocessed.
            If True, runs `fit_preprocessor` / `transform_preprocessor` per fold.
        preprocess_kwargs: optional dict of keyword arguments passed to fit_preprocessor

    Returns:
        results: dict containing:
            'best_gamma': float
            'best_tau': float
            'best_f1': float
            'mean_f1_grid': 2D array of shape (len(gammas), len(taus))
            'mean_acc_grid': 2D array of shape (len(gammas), len(taus))
            'mean_val_loss': 1D array of shape (len(gammas),)
    """
    if taus is None:
        taus = [0.5]
    gammas = list(gammas)
    taus = list(taus)
    preprocess_kwargs = preprocess_kwargs or {}

    k_indices = build_k_indices(y, k_fold, seed=seed)

    print(f"Preparing {k_fold} folds...")
    folds_data = []
    for k in range(k_fold):
        val_idx = k_indices[k]
        train_idx = np.delete(k_indices, k, axis=0).reshape(-1)

        x_tr_fold, x_val_fold = tx[train_idx], tx[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if use_full_preprocessor:
            params = fit_preprocessor(x_tr_fold, **preprocess_kwargs)
            tx_tr = transform_preprocessor(x_tr_fold, params)
            tx_val = transform_preprocessor(x_val_fold, params)
        else:
            tx_tr = x_tr_fold
            tx_val = x_val_fold

        y_tr_01 = np.where(y_tr == -1, 0.0, 1.0)
        y_val_01 = np.where(y_val == -1, 0.0, 1.0)

        folds_data.append((tx_tr, y_tr_01, tx_val, y_val, y_val_01))

    f1_grid = np.zeros((len(gammas), len(taus), k_fold))
    acc_grid = np.zeros((len(gammas), len(taus), k_fold))
    val_losses = np.zeros((len(gammas), k_fold))

    for g_idx, gamma in enumerate(gammas):
        print(f"\n--- Cross-validating gamma = {gamma} ---")
        for k, (tx_tr, y_tr_01, tx_val, y_val, y_val_01) in enumerate(folds_data):
            initial_w = np.zeros(tx_tr.shape[1])
            w, _ = logistic_regression(y_tr_01, tx_tr, initial_w, max_iters, gamma)

            # Compute validation probabilities once using existing sigmoid
            probs_val = np.clip(sigmoid(tx_val @ w), 1e-15, 1.0 - 1e-15)
            val_loss = -np.mean(
                y_val_01 * np.log(probs_val) + (1.0 - y_val_01) * np.log(1.0 - probs_val)
            )
            val_losses[g_idx, k] = val_loss

            # Sweep all candidate decision thresholds tau without retraining
            for t_idx, tau in enumerate(taus):
                y_pred = np.where(probs_val >= tau, 1, -1)
                f1_grid[g_idx, t_idx, k] = compute_f1(y_val, y_pred)
                acc_grid[g_idx, t_idx, k] = compute_accuracy(y_val, y_pred)

        mean_loss_g = np.mean(val_losses[g_idx])
        best_t_for_g = int(np.argmax(np.mean(f1_grid[g_idx], axis=1)))
        print(
            f"gamma={gamma}: mean val loss={mean_loss_g:.4f} | "
            f"best tau={taus[best_t_for_g]:.2f} -> mean F1={np.mean(f1_grid[g_idx, best_t_for_g]):.4f}, "
            f"mean Acc={np.mean(acc_grid[g_idx, best_t_for_g]):.4f}"
        )

    mean_f1_grid = np.mean(f1_grid, axis=2)
    mean_acc_grid = np.mean(acc_grid, axis=2)
    mean_val_loss = np.mean(val_losses, axis=1)

    best_g_idx, best_t_idx = np.unravel_index(np.argmax(mean_f1_grid), mean_f1_grid.shape)
    best_gamma = gammas[best_g_idx]
    best_tau = taus[best_t_idx]
    best_f1 = float(mean_f1_grid[best_g_idx, best_t_idx])

    return {
        "best_gamma": best_gamma,
        "best_tau": best_tau,
        "best_f1": best_f1,
        "mean_f1_grid": mean_f1_grid,
        "mean_acc_grid": mean_acc_grid,
        "mean_val_loss": mean_val_loss,
    }


def cross_validate_reg_logistic_regression(
    y,
    tx,
    gammas,
    lambdas,
    taus=None,
    k_fold=5,
    max_iters=500,
    seed=42,
    use_full_preprocessor=False,
    preprocess_kwargs=None,
):
    """Jointly cross-validate step-size (gamma), regularizer (lambda_), and threshold (tau)
    for regularized logistic regression (`reg_logistic_regression`).

    Args:
        y: 1D numpy array of labels in {-1, 1}, shape (N,)
        tx: 2D numpy array of features, shape (N, D). Can be already pre-processed
            (when use_full_preprocessor=False, as in run3.py) or raw (when use_full_preprocessor=True).
        gammas: list/1D array of candidate step sizes (gamma)
        lambdas: list/1D array of candidate regularization parameters (lambda_)
        taus: list/1D array of candidate decision thresholds in (0, 1). Defaults to [0.5] if None.
        k_fold: int, number of CV folds
        max_iters: int, number of gradient descent iterations
        seed: int, random seed for fold splitting
        use_full_preprocessor: bool. If False (default), uses `tx` directly as already preprocessed.
            If True, runs `fit_preprocessor` / `transform_preprocessor` per fold.
        preprocess_kwargs: optional dict of keyword arguments passed to fit_preprocessor

    Returns:
        results: dict containing:
            'best_gamma': float
            'best_lambda': float
            'best_tau': float
            'best_f1': float
            'mean_f1_grid': 3D array of shape (len(gammas), len(lambdas), len(taus))
            'mean_acc_grid': 3D array of shape (len(gammas), len(lambdas), len(taus))
            'mean_val_loss': 2D array of shape (len(gammas), len(lambdas))
    """
    if taus is None:
        taus = [0.5]
    gammas = list(gammas)
    lambdas = list(lambdas)
    taus = list(taus)
    preprocess_kwargs = preprocess_kwargs or {}

    k_indices = build_k_indices(y, k_fold, seed=seed)

    print(f"Preparing {k_fold} folds...")
    folds_data = []
    for k in range(k_fold):
        val_idx = k_indices[k]
        train_idx = np.delete(k_indices, k, axis=0).reshape(-1)

        x_tr_fold, x_val_fold = tx[train_idx], tx[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if use_full_preprocessor:
            params = fit_preprocessor(x_tr_fold, **preprocess_kwargs)
            tx_tr = transform_preprocessor(x_tr_fold, params)
            tx_val = transform_preprocessor(x_val_fold, params)
        else:
            tx_tr = x_tr_fold
            tx_val = x_val_fold

        y_tr_01 = np.where(y_tr == -1, 0.0, 1.0)
        y_val_01 = np.where(y_val == -1, 0.0, 1.0)

        folds_data.append((tx_tr, y_tr_01, tx_val, y_val, y_val_01))

    f1_grid = np.zeros((len(gammas), len(lambdas), len(taus), k_fold))
    acc_grid = np.zeros((len(gammas), len(lambdas), len(taus), k_fold))
    val_losses = np.zeros((len(gammas), len(lambdas), k_fold))

    for g_idx, gamma in enumerate(gammas):
        for l_idx, lambda_ in enumerate(lambdas):
            print(f"\n--- Cross-validating gamma = {gamma}, lambda_ = {lambda_} ---")
            for k, (tx_tr, y_tr_01, tx_val, y_val, y_val_01) in enumerate(folds_data):
                initial_w = np.zeros(tx_tr.shape[1])
                w, _ = reg_logistic_regression(
                    y_tr_01, tx_tr, lambda_, initial_w, max_iters, gamma
                )

                # Compute validation probabilities once using existing sigmoid
                probs_val = np.clip(sigmoid(tx_val @ w), 1e-15, 1.0 - 1e-15)
                val_loss = -np.mean(
                    y_val_01 * np.log(probs_val) + (1.0 - y_val_01) * np.log(1.0 - probs_val)
                )
                val_losses[g_idx, l_idx, k] = val_loss

                # Sweep all candidate decision thresholds tau without retraining
                for t_idx, tau in enumerate(taus):
                    y_pred = np.where(probs_val >= tau, 1, -1)
                    f1_grid[g_idx, l_idx, t_idx, k] = compute_f1(y_val, y_pred)
                    acc_grid[g_idx, l_idx, t_idx, k] = compute_accuracy(y_val, y_pred)

            mean_loss_gl = np.mean(val_losses[g_idx, l_idx])
            best_t_for_gl = int(np.argmax(np.mean(f1_grid[g_idx, l_idx], axis=1)))
            print(
                f"gamma={gamma}, lambda_={lambda_}: mean val loss={mean_loss_gl:.4f} | "
                f"best tau={taus[best_t_for_gl]:.2f} -> "
                f"mean F1={np.mean(f1_grid[g_idx, l_idx, best_t_for_gl]):.4f}, "
                f"mean Acc={np.mean(acc_grid[g_idx, l_idx, best_t_for_gl]):.4f}"
            )

    mean_f1_grid = np.mean(f1_grid, axis=3)
    mean_acc_grid = np.mean(acc_grid, axis=3)
    mean_val_loss = np.mean(val_losses, axis=2)

    best_g_idx, best_l_idx, best_t_idx = np.unravel_index(
        np.argmax(mean_f1_grid), mean_f1_grid.shape
    )
    best_gamma = gammas[best_g_idx]
    best_lambda = lambdas[best_l_idx]
    best_tau = taus[best_t_idx]
    best_f1 = float(mean_f1_grid[best_g_idx, best_l_idx, best_t_idx])

    return {
        "best_gamma": best_gamma,
        "best_lambda": best_lambda,
        "best_tau": best_tau,
        "best_f1": best_f1,
        "mean_f1_grid": mean_f1_grid,
        "mean_acc_grid": mean_acc_grid,
        "mean_val_loss": mean_val_loss,
    }
