"""CDC loading adapter, F1 threshold selection, and batched prediction."""
import csv
import numpy as np
from utils.helpers import load_csv_data


def load_data(folder):
    """Use the course loader, adding CDC feature names and label-ID alignment."""
    with (folder / 'x_train.csv').open() as file:
        names = next(csv.reader(file))[1:]
    with (folder / 'x_test.csv').open() as file:
        test_names = next(csv.reader(file))[1:]
    if names != test_names:
        raise ValueError('Training/test feature names or order differ')

    x_train, x_test, y, train_ids, test_ids = load_csv_data(folder)
    label_ids = np.genfromtxt(folder / 'y_train.csv', delimiter=',', skip_header=1,
                              usecols=0, dtype=int, ndmin=1)
    if len(train_ids) != len(label_ids):
        raise ValueError('Training and label row counts differ')
    for ids in (train_ids, test_ids, label_ids):
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate IDs in a data file')
    order = np.argsort(label_ids)
    positions = np.searchsorted(label_ids[order], train_ids)
    if np.any(positions >= len(y)) or not np.array_equal(label_ids[order[positions]], train_ids):
        raise ValueError('Training IDs and label IDs differ')
    y = y[order[positions]]
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError('Expected both -1 and +1 labels')
    if x_train.shape[1] != len(names) or x_test.shape[1] != len(names):
        raise ValueError('Unexpected feature columns')
    return x_train, y, x_test, test_ids, names


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
