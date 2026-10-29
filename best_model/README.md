# Best model: class-weighted ridge

From the repository root, with the project environment activated:

```sh
python -m best_model.run
```

Or run `python best_model/run.py`. Put the original `x_train.csv`, `y_train.csv`
and `x_test.csv` in the repository root's `data/` directory, with training rows
and labels in their original matching order. Python and NumPy are required.

The runner uses the shared root `implementations.py` ridge function and
`utils/helpers.py` course loader and CSV writer. It fits CDC preprocessing on
training rows, applies mean-normalized positive-class weight 4 (negative weight
1), and trains ridge with lambda 0.001. Training rows and labels are multiplied
by square-root weights; predictions use ordinary CDC features. The threshold
maximizes training F1. Test data never fits preprocessing or selects the threshold.

Output is `best_model/submission_shared_ridge.csv`; repeated runs add a number
and preserve existing outputs. Training requires several GB of memory. The
script does not upload the CSV.

Files:
- `run.py`: load, preprocess, train, choose threshold and write predictions.
- `preprocessing.py`: CDC encoding, numeric imputation/standardization and weighting.
- `preprocessing_rules.json`: CDC feature definitions.
- `submission_helpers.py`: F1 threshold selection and batched prediction.

The original weighted-ridge submission scored F1 0.445 and accuracy 0.879.
This shared-ridge implementation is mathematically equivalent; exact CSV
reproduction after refactoring still needs a full-data check.
