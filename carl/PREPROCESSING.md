# Shared CDC preprocessing

Use `carl/preprocessing.py` together with `carl/preprocessing_rules.json`.
Dependencies: NumPy and the Python standard library. The preprocessor does not
import any model implementation or require Carl's experiment scripts.

Run Python from the repository root so `carl` can be imported.

```python
import csv
from carl.preprocessing import CDCPreprocessor

# x_train/x_test contain the raw 321 feature columns, without Id.
# Missing CSV entries should be loaded as np.nan.
with open("data/x_train.csv") as file:
    train_names = next(csv.reader(file))[1:]
with open("data/x_test.csv") as file:
    test_names = next(csv.reader(file))[1:]

preprocessor = CDCPreprocessor()
tx_train = preprocessor.fit_transform(x_train, train_names)
tx_test = preprocessor.transform(x_test, test_names)

# Feed tx_train to your model and use tx_test for predictions.
```

The output has 2,150 columns, including a leading intercept of ones. Do not add
another intercept. Output uses float32; a full expanded training matrix needs
about 2.8 GB, plus input and temporary arrays. For lower-memory test predictions,
use `preprocessor.transform_batches(x_test, test_names)` after fitting your model.
Keep the fitted preprocessor with your weights so test data uses training statistics.

The preprocessing:

- Excludes 31 administrative/design features and HAREHAB1.
- Interprets response codes separately for each feature using reviewed CDC rules.
- Expands categorical responses, including unknown/refused/blank categories.
- Decodes numerical units and special formats, fills missing numerical values
  with training medians, and adds missing/reason indicators.
- Standardizes numerical outputs with training means/scales; flags remain 0/1.

It preserves row order, does not change labels, and does not remove participants.
The input names and their order must match the original course dataset.

For cross-validation, create a fresh `CDCPreprocessor` for every fold. Fit it only
on that fold's training rows, then transform validation rows with that same
object. Never fit preprocessing on validation or test data.

Models keep their own label conventions: MSE uses -1/+1; the shared logistic
implementation expects 0/1. Convert labels separately when needed.
