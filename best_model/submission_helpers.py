"""F1 threshold selection and batched prediction."""
import numpy as np


def training_threshold(y, predictions):
    """Maximize training F1; include tied scores together, prefer higher cutoffs."""
    order = np.argsort(-predictions, kind='stable')
    values = predictions[order]
    ends = np.r_[np.flatnonzero(np.diff(values) != 0), len(values)-1]
    true_positives = np.cumsum(y[order] == 1)[ends]
    f1 = 2 * true_positives / (ends + 1 + np.sum(y == 1))
    return float(values[ends[np.argmax(f1)]])


def predict_scores(preprocessor, raw, names, w):
    """Predict with ordinary CDC features in batches to limit memory use."""
    if not np.isfinite(w).all():
        raise FloatingPointError('Ridge returned nonfinite coefficients')
    scores = np.empty(len(raw), dtype=np.float64)
    start = 0
    for batch in preprocessor.transform_batches(raw, names):
        scores[start:start + len(batch)] = batch.astype(np.float64) @ w
        start += len(batch)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Predictions contain nonfinite values')
    return scores
