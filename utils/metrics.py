import numpy as np

def compute_accuracy(y_true, y_pred):
    """Computes classification accuracy."""
    return np.mean(y_true == y_pred)

def compute_precision(y_true, y_pred):
    """Computes binary precision score."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def compute_recall(y_true, y_pred):
    """Computes binary recall score."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def compute_f1(y_true, y_pred):
    """Computes binary F1 score."""
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_r2(y_true, y_pred):
    """Computes R^2 (coefficient of determination) regression score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)
