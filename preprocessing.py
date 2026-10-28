"""CDC-aware preprocessing for the course's 2015 BRFSS CSV files.

Only NumPy and the standard library are required. Rules are stored beside this
module. Fit on training rows only; reuse the fitted object for validation/test.
"""

import json
from pathlib import Path

import numpy as np


class CDCPreprocessor:
    """Fixed codebook encoding, training medians/scales, and an intercept.

    Output is float32 to limit memory use; calculations and fitted statistics
    use float64. No rows are dropped, and input arrays are never modified.
    All 321 named source columns are required, including excluded columns.
    """

    def __init__(self):
        path = Path(__file__).with_name("preprocessing_rules.json")
        self.rules = json.loads(path.read_text())["rules"]
        self.input_features = tuple(rule["name"] for rule in self.rules)
        self.excluded_features = tuple(r["name"] for r in self.rules if r["kind"] == "drop")
        self.statistics_ = None
        self.feature_names_ = ("intercept",) + tuple(
            name for name, _, _ in self._columns(np.empty((0, len(self.rules))))
        )
        if len(self.feature_names_) != len(set(self.feature_names_)):
            raise ValueError("Duplicate output names in preprocessing rules")

    def _validate(self, x, feature_names):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != len(self.input_features):
            raise ValueError("Expected a matrix containing the 321 original features (no Id)")
        if tuple(feature_names) != self.input_features:
            raise ValueError("Input feature names/order differ from the reviewed schema")
        if x.dtype.kind not in "fiu" or np.isinf(x).any():
            raise ValueError("Inputs must be numeric, with NaN for blanks and no infinities")
        return x

    def fit(self, x_train, feature_names):
        """Learn numerical medians, means and scales from training rows only."""
        x_train = self._validate(x_train, feature_names)
        if len(x_train) == 0:
            raise ValueError("Cannot fit on an empty training split")
        statistics = {}
        # Only numerical outputs have fitted parameters; categorical levels are
        # fixed by the independent codebook, including absent training levels.
        for name, values, indicator in self._columns(x_train, numerical_only=True):
            if indicator:
                continue
            valid = np.isfinite(values)
            median = float(np.median(values[valid])) if valid.any() else 0.0
            filled = np.where(valid, values, median)
            mean = float(filled.mean())
            scale = float(filled.std())
            if scale < 1e-12:
                scale = 1.0
            statistics[name] = (median, mean, scale)
        self.statistics_ = statistics
        return self

    def transform(self, x, feature_names):
        """Apply fixed columns and training statistics without refitting."""
        if self.statistics_ is None:
            raise ValueError("Call fit on training rows before transform")
        x = self._validate(x, feature_names)
        result = np.empty((len(x), len(self.feature_names_)), dtype=np.float32)
        result[:, 0] = 1.0
        for j, (name, values, indicator) in enumerate(self._columns(x), 1):
            if name != self.feature_names_[j]:
                raise RuntimeError("Output schema changed")
            if not indicator:
                median, mean, scale = self.statistics_[name]
                values = (np.where(np.isfinite(values), values, median) - mean) / scale
            result[:, j] = values
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced nonfinite output")
        return result

    def fit_transform(self, x_train, feature_names):
        """Fit once and return the transformed training matrix."""
        return self.fit(x_train, feature_names).transform(x_train, feature_names)

    def transform_batches(self, x, feature_names, batch_size=10000):
        """Yield transformed rows in order without allocating the full test matrix."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.statistics_ is None:
            raise ValueError("Call fit before transform_batches")
        x = self._validate(x, feature_names)
        for start in range(0, len(x), batch_size):
            yield self.transform(x[start:start + batch_size], feature_names)

    @staticmethod
    def _category_columns(name, raw, rule):
        values = raw.copy()
        if rule.get("group_six_plus"):
            mask = np.isfinite(values) & (values == np.floor(values)) & (values >= 6) & (values <= 99)
            values[mask] = 6
        known = np.zeros(len(values), dtype=bool)
        for code in rule["categories"]:
            mask = values == code
            known |= mask
            yield f"{name}__code_{code}", mask, True
        yield name + "__blank", np.isnan(raw), True
        yield name + "__unrecognized", ~np.isnan(raw) & ~known, True

    @staticmethod
    def _number_columns(name, values, flags):
        yield name, values, False
        yield name + "__unavailable", ~np.isfinite(values), True
        for reason, mask in sorted(flags.items()):
            yield name + "__" + reason, mask, True

    @staticmethod
    def _decode_numeric(raw, rule, parents):
        values = raw.astype(np.float64, copy=True)
        flags = {"blank": np.isnan(raw)}
        missing = np.zeros(len(raw), dtype=bool)
        for code, reason in rule["missing_codes"].items():
            mask = raw == float(code)
            flags[reason] = flags.get(reason, np.zeros(len(raw), dtype=bool)) | mask
            missing |= mask
        limit = rule.get("reject_unconverted_units_above")
        if limit is not None and np.any(np.isfinite(raw) & ~missing & (raw > limit)):
            raise ValueError(f"{rule['name']}: values suggest a different unit/storage format; expected the reviewed course CSV units")
        base = np.zeros(len(raw), dtype=bool)
        for low, high in rule["valid_ranges"]:
            base |= (raw >= low) & (raw <= high)
        if rule["integer"]:
            base &= raw == np.floor(raw)
        base &= ~missing & np.isfinite(raw)
        codec = rule["codec"]
        safe = np.where(base, raw, 0.0)
        prefix, count = np.floor(safe / 100), safe % 100
        if codec == "food_frequency":
            values = np.where(prefix == 1, count, np.where(prefix == 2, count / 7, count / 30))
        elif codec == "weekly_frequency":
            values = np.where(prefix == 1, count, count * 7 / 30)
        elif codec == "alcohol_days":
            values = np.where(prefix == 1, count / 7, count / 30)
        elif codec == "daily_frequency":
            values = count / np.select([prefix == 1, prefix == 2, prefix == 3], [1., 7., 30.], default=365.)
        elif codec == "duration_days":
            values = count * np.select([prefix == 1, prefix == 2, prefix == 3], [1., 7., 30.], default=365.)
        elif codec == "hours_minutes":
            base &= count < 60
            values = prefix * 60 + count
        elif codec == "weight_kg":
            values = np.where(safe >= 9000, safe - 9000, safe * .4535924)
            base &= values > 0
        elif codec == "height_metres":
            base &= (safe >= 9000) | (count < 12)
            values = np.where(safe >= 9000, (safe - 9000) / 100, (prefix * 12 + count) * .0254)
            base &= values > 0
        elif codec != "identity":
            raise ValueError(f"Unknown codec: {codec}")
        available = base.copy()
        for code, replacement in rule["recode"].items():
            mask = raw == float(code)
            values[mask] = replacement
            available |= mask
        if "confirm_99_with" in rule:
            ambiguous = (raw == 99) & (parents[rule["confirm_99_with"]] != 199)
            flags["ambiguous"] = ambiguous
            available &= ~ambiguous
        else:
            ambiguous = np.zeros(len(raw), dtype=bool)
        flags["invalid"] = ~np.isnan(raw) & ~missing & ~available & ~ambiguous
        for reason, codes in rule["value_flags"].items():
            flags[reason] = np.isin(raw, codes) & available
        values[~available] = np.nan
        values /= rule["divisor"]
        return values, flags

    @staticmethod
    def _date_columns(name, raw, rule, numerical_only):
        blank = np.isnan(raw)
        unknown, refused = raw == 777777, raw == 999999
        safe = np.where(np.isfinite(raw), raw, 0)
        integer = safe == np.floor(safe)
        month, year = np.floor(safe / 10000), safe % 10000
        ordinary = ~blank & ~unknown & ~refused & integer
        month_known = ordinary & (month >= 1) & (month <= 12)
        month_unknown = ordinary & (month == 77) & rule["allow_unknown_month"]
        low, high = rule["year_range"]
        valid_year = (month_known | month_unknown) & (year >= low) & (year <= high)
        flags = {"blank": blank, "unknown": unknown, "refused": refused,
                 "invalid": ~blank & ~unknown & ~refused & ~valid_year}
        yield from CDCPreprocessor._number_columns(name + "__year", np.where(valid_year, year, np.nan), flags)
        if not numerical_only:
            for code in range(1, 13):
                yield f"{name}__month_{code}", month_known & (month == code), True
            for reason, mask in {"unknown": unknown | month_unknown, "refused": refused, "blank": blank,
                                 "unrecognized": ~blank & ~unknown & ~refused & ~month_known & ~month_unknown}.items():
                yield name + "__month_" + reason, mask, True

    @staticmethod
    def _multiple_columns(name, raw):
        selected = np.zeros((len(raw), 4), dtype=bool)
        unknown, refused = raw == 7, raw == 9
        blank = np.isnan(raw)
        recognized = unknown | refused | blank
        # There are very few distinct answer combinations; parse each once.
        for value in np.unique(raw[np.isfinite(raw)]):
            if value != np.floor(value) or value < 1 or value > 9999:
                continue
            digits = set(str(int(value)))
            if digits <= set("12348"):
                mask = raw == value
                recognized |= mask
                for j in range(4):
                    if str(j + 1) in digits:
                        selected[mask, j] = True
        for j in range(4):
            yield f"{name}__person_{j+1}", selected[:, j], True
        for reason, mask in {"unknown": unknown, "refused": refused, "blank": blank,
                             "unrecognized": ~recognized, "no_additional_choices": raw == 8}.items():
            yield name + "__" + reason, mask, True

    def _columns(self, x, numerical_only=False):
        parents = dict(zip(self.input_features, x.T))
        for rule in self.rules:
            name, kind = rule["name"], rule["kind"]
            raw = parents[name]
            if kind == "drop":
                continue
            if kind == "numeric":
                values, flags = self._decode_numeric(raw, rule, parents)
                yield from self._number_columns(name, values, flags)
            elif kind == "date":
                yield from self._date_columns(name, raw, rule, numerical_only)
            elif not numerical_only and kind == "categorical":
                yield from self._category_columns(name, raw, rule)
            elif not numerical_only and kind == "multiple_answers":
                yield from self._multiple_columns(name, raw)


def weight_training_data(tx, y, positive_weight=4.0):
    """Scale training features AND labels for weighted fitting with ordinary ridge.

    Positive examples get `positive_weight`, negatives get 1. We normalize the
    weights to mean 1 so lambda retains the same meaning as in weighted ridge.
    Only call this for fitting. Threshold selection and test predictions must
    use ordinary CDC-preprocessed rows. Inputs are not modified.
    """
    y = np.asarray(y)
    if tx.ndim != 2 or y.shape != (len(tx),) or not len(y):
        raise ValueError('Expected nonempty training features (N,D) and labels (N,)')
    if not np.isin(y, [-1, 1]).all():
        raise ValueError('Expected labels -1/+1')
    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError('positive_weight must be positive and finite')
    weights = np.where(y == 1, positive_weight, 1.)
    factors = np.sqrt(weights / weights.mean())
    weighted_tx = tx.astype(np.float64, copy=True)
    weighted_tx *= factors[:, None]
    return weighted_tx, y.astype(np.float64) * factors
