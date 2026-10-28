"""Train our best weighted ridge model and save its submission CSV."""
from pathlib import Path
import numpy as np
from submission_helpers import load_data, predict_scores, training_threshold
from utils.helpers import create_csv_submission
from preprocessing import CDCPreprocessor, weight_training_data

from implementations import ridge_regression

PROJECT = Path(__file__).resolve().parent
POSITIVE_WEIGHT = 4.0
LAMBDA = 0.001
OUTPUT = PROJECT / 'submission_shared_ridge.csv'


def main():
    output = OUTPUT
    number = 2
    while output.exists():
        output = OUTPUT.with_name(f'{OUTPUT.stem}_{number}{OUTPUT.suffix}')
        number += 1

    # --- LOAD DATA ---
    print('Loading data...', flush=True)
    x_train, y_train, x_test, test_ids, names = load_data(PROJECT / 'data')

    # --- PREPROCESSING ---
    print('Preparing CDC features...', flush=True)
    preprocessor = CDCPreprocessor()
    x_weighted = preprocessor.fit_transform(x_train, names)
    x_weighted, y_weighted = weight_training_data(x_weighted, y_train, POSITIVE_WEIGHT)

    # --- TRAINING ---
    print('Training weighted ridge...', flush=True)
    w, loss = ridge_regression(y_weighted, x_weighted, LAMBDA)
    del x_weighted, y_weighted  # Free memory before predicting.

    # --- SELECT F1 THRESHOLD (ordinary features, without training weights) ---
    train_scores = predict_scores(preprocessor, x_train, names, w)
    threshold = training_threshold(y_train, train_scores)

    # --- PREDICT AND SAVE ---
    test_scores = predict_scores(preprocessor, x_test, names, w)
    predictions = np.where(test_scores >= threshold, 1, -1)
    create_csv_submission(test_ids, predictions, output)
    print(f'Weighted training MSE: {loss:.6f}; threshold: {threshold:.12f}')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
