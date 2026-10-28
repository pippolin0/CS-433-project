"""CSV loading, threshold selection, and course-format submission writing."""
import csv
import numpy as np


def load_data(folder):
    with (folder / 'x_train.csv').open() as file:
        names = next(csv.reader(file))[1:]
    with (folder / 'x_test.csv').open() as file:
        test_names = next(csv.reader(file))[1:]
    if names != test_names:
        raise ValueError('Training/test feature names or order differ')
    def read(name):
        return np.loadtxt(folder / name, delimiter=',', skiprows=1, ndmin=2,
                          converters=lambda value: float(value) if value else np.nan)
    raw = read('x_train.csv')
    labels = read('y_train.csv')
    test = read('x_test.csv')
    if raw.shape[1] != len(names)+1 or test.shape[1] != len(names)+1 or labels.shape[1] != 2:
        raise ValueError('Unexpected CSV columns')
    for array in (raw, labels, test):
        ids = array[:, 0]
        if (not len(ids) or not np.isfinite(ids).all() or not np.all(ids == np.floor(ids))
                or len(np.unique(ids)) != len(ids)):
            raise ValueError('IDs must be unique finite integers within each CSV')
    if len(raw) != len(labels):
        raise ValueError('Training and label row counts differ')
    order = np.argsort(labels[:, 0])
    positions = np.searchsorted(labels[order, 0], raw[:, 0])
    if np.any(positions >= len(labels)) or not np.array_equal(labels[order[positions], 0], raw[:, 0]):
        raise ValueError('Training IDs and label IDs differ')
    y = labels[order[positions], 1]
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError('Expected both -1 and +1 labels')
    if np.isinf(raw[:, 1:]).any() or np.isinf(test[:, 1:]).any():
        raise ValueError('Features must not contain infinity; NaN denotes missing values')
    return raw[:, 1:], y, test[:, 1:], test[:, 0], names


def training_threshold(y, predictions):
    """Maximize training F1; include tied scores together, prefer higher cutoffs."""
    order = np.argsort(-predictions, kind='stable')
    values = predictions[order]
    ends = np.r_[np.flatnonzero(np.diff(values) != 0), len(values)-1]
    true_positives = np.cumsum(y[order] == 1)[ends]
    f1 = 2 * true_positives / (ends + 1 + np.sum(y == 1))
    return float(values[ends[np.argmax(f1)]])


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


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
