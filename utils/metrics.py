# import numpy as np


# def compute_accuracy(y_true, y_pred):
#     """Computes classification accuracy."""
#     return np.mean(y_true == y_pred)

# def compute_precision(y_true, y_pred):
#     """Computes binary precision score."""
#     tp = np.sum((y_true == 1) & (y_pred == 1))
#     fp = np.sum((y_true == -1) & (y_pred == 1))
#     return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# def compute_recall(y_true, y_pred):
#     """Computes binary recall score."""
#     tp = np.sum((y_true == 1) & (y_pred == 1))
#     fn = np.sum((y_true == 1) & (y_pred == -1))
#     return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# def compute_f1(y_true, y_pred):
#     """Computes binary F1 score."""
#     precision = compute_precision(y_true, y_pred)
#     recall = compute_recall(y_true, y_pred)
#     if precision + recall == 0:
#         return 0.0
#     return 2 * (precision * recall) / (precision + recall)

# def compute_r2(y_true, y_pred):
#     """Computes R^2 (coefficient of determination) regression score."""
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     if ss_tot == 0:
#         return 0.0
#     return 1.0 - (ss_res / ss_tot)

import numpy as np

def _to_binary(y_pred):
    """Converts continuous predictions/probabilities into binary 0 or 1 labels."""
    # Threshold at 0.5 if probabilities in [0, 1], or threshold at 0.0 if raw logits
    if np.min(y_pred) < 0:
        return (y_pred > 0).astype(int)
    return (y_pred >= 0.5).astype(int)

def compute_accuracy(y_true, y_pred):
    y_pred_bin = _to_binary(y_pred)
    y_true_bin = (y_true > 0).astype(int)
    return np.mean(y_true_bin.ravel() == y_pred_bin.ravel())

def compute_precision(y_true, y_pred):
    y_pred_bin = _to_binary(y_pred)
    y_true_bin = (y_true > 0).astype(int)
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def compute_recall(y_true, y_pred):
    y_pred_bin = _to_binary(y_pred)
    y_true_bin = (y_true > 0).astype(int)
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def compute_f1(y_true, y_pred):
    p = compute_precision(y_true, y_pred)
    r = compute_recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_r2(y_true, y_pred):
    """Computes the R^2 (coefficient of determination) regression score."""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    
    if ss_tot == 0:
        return 0.0  # Avoid division by zero if all target values are constant
    
    return 1.0 - (ss_res / ss_tot)