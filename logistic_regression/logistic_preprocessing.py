# -*- coding: utf-8 -*-
"""Modular pure-NumPy data preprocessing pipeline for the BRFSS dataset.

Implements:
1. Automated Column Type Detector (constant, binary, categorical, continuous)
2. Selective BRFSS Sentinel Code Cleaner for continuous columns (only those that need it)
3. Fit / Transform Preprocessing Pattern (preventing data leakage during K-fold CV and test evaluation)
"""

import numpy as np


# Administrative / survey metadata column indices (0-indexed after removing Id):
# 1: FMONTH, 2: IDATE, 3: IMONTH, 4: IDAY, 5: IYEAR, 6: DISPCODE,
# 7: SEQNO, 8: _PSU, 9: CTELENUM, 18: CTELNUM1, 217: QSTVER, 218: QSTLANG
DEFAULT_METADATA_INDICES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 217, 218)


def _detect_continuous_sentinel_rules(col_valid):
    """Detect whether a continuous column contains BRFSS sentinel codes.

    Only flags sentinel codes when the column's empirical distribution shows
    the structural signature of BRFSS coding (e.g., isolated sentinel values at
    the upper tail with a gap separating them from valid measurements), ensuring
    genuine continuous measurements (such as age 77 in _AGE80 or height 77 in HTIN4)
    are never modified.

    Args:
        col_valid: 1D numpy array of non-NaN values from a single training column.

    Returns:
        dict with keys:
            'zero_codes': list of values to map to 0.0 (e.g. 88, 888, 555 meaning 'None')
            'nan_codes': list of values to map to np.nan (e.g. 77, 99, 777, 999 meaning 'Don't know / Refused')
            'decode_freq': bool indicating whether 1xx/2xx/3xx/4xx frequency encoding is used
            'decode_weight_height': str or None ('weight' or 'height')
    """
    rules = {
        "zero_codes": [],
        "nan_codes": [],
        "decode_freq": False,
        "decode_weight_height": None,
    }
    if col_valid.size == 0:
        return rules

    u = np.unique(col_valid)
    max_val = u[-1]
    min_val = u[0]
    is_all_int = np.all(np.floor(u) == u)

    # Case 1: 6-digit date/month-year sentinels (e.g. FLSHTMY2, HIVTSTD3: 777777, 999999)
    if is_all_int and max_val in (777777.0, 999999.0):
        rules["nan_codes"].extend([777777.0, 999999.0])
        return rules

    # Case 2: 5-digit drinking-per-week sentinel (_DRNKWEK: 99900 = Don't know/Refused)
    if is_all_int and max_val == 99900.0:
        rules["nan_codes"].append(99900.0)
        return rules

    # Case 3: 4-digit WEIGHT2 / HEIGHT3 columns (7777, 9999 = Don't know/Refused, 9xxx = metric)
    if is_all_int and max_val in (7777.0, 9999.0) and min_val >= 50.0:
        rules["nan_codes"].extend([7777.0, 9999.0])
        if min_val < 300.0 and np.any((u >= 50.0) & (u < 100.0)):
            rules["decode_weight_height"] = "weight"
        elif min_val >= 200.0 and np.any((u >= 400.0) & (u <= 711.0)):
            rules["decode_weight_height"] = "height"
        return rules

    # Case 4: 3-digit sentinels (555, 777, 888, 999 or 900)
    if max_val in (777.0, 888.0, 999.0, 900.0):
        non_sentinel = u[~np.isin(u, [555.0, 777.0, 888.0, 900.0, 999.0])]
        # Only treat as sentinel if all normal values are well below 777 (or 900)
        if non_sentinel.size > 0 and non_sentinel[-1] <= 600.0:
            if 555.0 in u:
                rules["zero_codes"].append(555.0)
            if 888.0 in u:
                rules["zero_codes"].append(888.0)
            for code in (777.0, 900.0, 999.0):
                if code in u:
                    rules["nan_codes"].append(code)
            # Check if this is a 1xx/2xx/3xx/4xx BRFSS frequency variable (e.g. ALCDAY5, FRUIT1)
            if is_all_int and non_sentinel[0] >= 101.0 and non_sentinel[-1] < 500.0:
                rules["decode_freq"] = True
            return rules

    # Case 5: 2-digit sentinels (77, 88, 97, 98, 99)
    # Must be integer-valued with max in {77, 88, 97, 98, 99} and a gap separating normal values
    # (This protects _AGE80 [18..80] and HTIN4 [48..82] where values are contiguous up to 80/82)
    if is_all_int and max_val in (77.0, 88.0, 97.0, 98.0, 99.0):
        candidate_sentinels = [77.0, 88.0, 96.0, 97.0, 98.0, 99.0]
        non_sentinel = u[~np.isin(u, candidate_sentinels)]
        # Verify there is a gap (normal values stop <= 76 and don't smoothly bridge 76->77->78->79)
        has_contiguous_70s_80s = all(v in u for v in (75.0, 76.0, 77.0, 78.0, 79.0, 80.0))
        if non_sentinel.size > 0 and not has_contiguous_70s_80s:
            if 88.0 in u and (87.0 not in u and 89.0 not in u):
                rules["zero_codes"].append(88.0)
            for code in (77.0, 96.0, 97.0, 98.0, 99.0):
                if code in u and (code - 1.0 not in non_sentinel):
                    rules["nan_codes"].append(code)

    return rules


def _apply_continuous_sentinel_rules(col, rules):
    """Apply the detected sentinel cleaning rules to a single 1D column copy."""
    col = col.astype(float, copy=True)

    for z_code in rules["zero_codes"]:
        col[col == z_code] = 0.0

    for n_code in rules["nan_codes"]:
        col[col == n_code] = np.nan

    # Convert BRFSS compound frequency codes (1xx = per day/week, 2xx = per week/month, 3xx = per month, 4xx = per year)
    # into a unified monthly frequency scale
    if rules["decode_freq"]:
        mask_100 = (col >= 101.0) & (col <= 199.0)
        mask_200 = (col >= 201.0) & (col <= 299.0)
        mask_300 = (col >= 301.0) & (col <= 399.0)
        mask_400 = (col >= 401.0) & (col <= 499.0)
        col[mask_100] = (col[mask_100] - 100.0) * 4.33  # times per week -> per month
        col[mask_200] = col[mask_200] - 200.0           # times per month
        col[mask_300] = col[mask_300] - 300.0           # times per month
        col[mask_400] = (col[mask_400] - 400.0) / 12.0  # times per year -> per month

    if rules["decode_weight_height"] == "weight":
        # 9000-9998 is kilograms; <1000 is pounds -> convert pounds to kg
        mask_metric = (col >= 9000.0) & (col <= 9998.0)
        mask_lbs = (col > 0.0) & (col < 1000.0)
        col[mask_metric] = col[mask_metric] - 9000.0
        col[mask_lbs] = col[mask_lbs] * 0.453592

    elif rules["decode_weight_height"] == "height":
        # 9000-9998 is cm; 200-711 is feet (hundreds digit) and inches (last two digits) -> convert to cm
        mask_metric = (col >= 9000.0) & (col <= 9998.0)
        mask_ft_in = (col >= 200.0) & (col <= 711.0)
        col[mask_metric] = col[mask_metric] - 9000.0
        feet = np.floor(col[mask_ft_in] / 100.0)
        inches = col[mask_ft_in] % 100.0
        col[mask_ft_in] = (feet * 12.0 + inches) * 2.54

    return col


def detect_column_types(
    x_train,
    max_nan_ratio=0.70,
    max_categorical_unique=10,
    drop_metadata_indices=DEFAULT_METADATA_INDICES,
):
    """Inspect training columns and classify each into dropped, binary, categorical, or continuous.

    Args:
        x_train: 2D numpy array of shape (N, D)
        max_nan_ratio: float in (0, 1], drop columns with a higher fraction of NaNs
        max_categorical_unique: int, max number of unique values for categorical columns
        drop_metadata_indices: iterable of column indices to drop immediately (survey IDs/dates)

    Returns:
        dict mapping column categories and per-column metadata.
    """
    n_samples, n_cols = x_train.shape
    drop_set = set(drop_metadata_indices) if drop_metadata_indices is not None else set()

    col_info = {}
    summary = {
        "dropped_metadata": [],
        "dropped_high_nan": [],
        "dropped_constant": [],
        "binary": [],
        "categorical": [],
        "continuous": [],
        "continuous_cleaned": [],
    }

    for j in range(n_cols):
        if j in drop_set:
            summary["dropped_metadata"].append(j)
            continue

        col = x_train[:, j]
        nan_ratio = np.mean(np.isnan(col))
        if nan_ratio > max_nan_ratio:
            summary["dropped_high_nan"].append(j)
            continue

        valid = col[~np.isnan(col)]
        u = np.unique(valid)

        if u.size <= 1:
            summary["dropped_constant"].append(j)
            continue

        # Check if this is a small integer-coded question where 7/9 (or 8) are sentinel codes
        # e.g. [1, 2, 7, 9] -> Yes(1), No(2), Don't know(7), Refused(9)
        small_sentinel_codes = []
        if u.size <= max_categorical_unique and np.all(np.floor(u) == u) and u[-1] <= 9.0:
            non_79 = u[~np.isin(u, [7.0, 9.0])]
            # If normal options are <= 5 (e.g. 1..2 or 1..5) and 6 is not present, 7 and 9 are sentinels
            if non_79.size >= 2 and non_79[-1] <= 5.0 and (6.0 not in u):
                for sc in (7.0, 9.0):
                    if sc in u:
                        small_sentinel_codes.append(sc)
                u = non_79

        if u.size <= 1:
            summary["dropped_constant"].append(j)
            continue

        if u.size == 2:
            # Binary column: map {u[0], u[1]} -> {0.0, 1.0}
            col_info[j] = {
                "type": "binary",
                "val_zero": float(u[0]),
                "val_one": float(u[1]),
                "nan_codes": small_sentinel_codes,
            }
            summary["binary"].append(j)

        elif u.size <= max_categorical_unique:
            # Categorical column (3 to max_categorical_unique unique values)
            zero_codes = [8.0] if (8.0 in u and 6.0 not in u and u[0] >= 1.0) else []
            u_clean = np.array([0.0 if v in zero_codes else v for v in u])
            u_clean = np.unique(u_clean)
            col_info[j] = {
                "type": "categorical",
                "categories": u_clean,
                "zero_codes": zero_codes,
                "nan_codes": small_sentinel_codes,
            }
            summary["categorical"].append(j)

        else:
            # Continuous column (> max_categorical_unique unique values)
            sentinel_rules = _detect_continuous_sentinel_rules(valid)
            needs_cleaning = bool(
                sentinel_rules["zero_codes"]
                or sentinel_rules["nan_codes"]
                or sentinel_rules["decode_freq"]
                or sentinel_rules["decode_weight_height"]
            )
            col_info[j] = {
                "type": "continuous",
                "sentinel_rules": sentinel_rules,
                "needs_cleaning": needs_cleaning,
            }
            summary["continuous"].append(j)
            if needs_cleaning:
                summary["continuous_cleaned"].append(j)

    return {"col_info": col_info, "summary": summary}


def fit_preprocessor(
    x_train,
    max_nan_ratio=0.70,
    max_categorical_unique=10,
    drop_metadata_indices=DEFAULT_METADATA_INDICES,
    one_hot_categorical=True,
    add_bias=True,
):
    """Fit the preprocessing parameters on x_train only (to prevent data leakage).

    Steps fitted on x_train:
    1. Detect column types (drop metadata/high-NaN/constant, separate binary, categorical, continuous).
    2. For binary columns: map {v0, v1} -> {0, 1}, compute mode in {0, 1} for NaN imputation.
    3. For categorical columns: clean 7/9 sentinels, compute mode for NaN imputation, and record
       categories for one-hot encoding (dropping the first category as reference to avoid collinearity)
       or mean/std if one_hot_categorical=False.
    4. For continuous columns: selectively clean BRFSS sentinel codes (only on columns that need it),
       compute column median for NaN imputation, and compute mean and std for Z-score standardization.

    Args:
        x_train: 2D numpy array of shape (N_train, D)
        max_nan_ratio: maximum allowed NaN fraction per column
        max_categorical_unique: threshold between categorical and continuous unique count
        drop_metadata_indices: column indices of administrative metadata to drop
        one_hot_categorical: if True, one-hot encode categorical features; else standardize them
        add_bias: if True, prepend a column of 1s during transform

    Returns:
        params: dict containing all fitted statistics needed by transform_preprocessor.
    """
    detection = detect_column_types(
        x_train,
        max_nan_ratio=max_nan_ratio,
        max_categorical_unique=max_categorical_unique,
        drop_metadata_indices=drop_metadata_indices,
    )
    col_info = detection["col_info"]
    summary = detection["summary"]

    fitted_cols = {}
    for j, info in col_info.items():
        col = x_train[:, j].astype(float, copy=True)
        ctype = info["type"]

        if ctype == "binary":
            for nc in info["nan_codes"]:
                col[col == nc] = np.nan
            mapped = np.full_like(col, np.nan)
            mapped[col == info["val_zero"]] = 0.0
            mapped[col == info["val_one"]] = 1.0
            # Mode in {0.0, 1.0}
            valid_mapped = mapped[~np.isnan(mapped)]
            mode_val = 1.0 if np.mean(valid_mapped) >= 0.5 else 0.0
            fitted_cols[j] = {
                **info,
                "impute_value": mode_val,
            }

        elif ctype == "categorical":
            for nc in info["nan_codes"]:
                col[col == nc] = np.nan
            for zc in info["zero_codes"]:
                col[col == zc] = 0.0
            valid = col[~np.isnan(col)]
            vals, counts = np.unique(valid, return_counts=True)
            mode_val = float(vals[np.argmax(counts)]) if vals.size > 0 else 0.0
            imputed = np.where(np.isnan(col), mode_val, col)
            mean_val = float(np.mean(imputed))
            std_val = float(np.std(imputed))
            if std_val == 0.0:
                std_val = 1.0
            fitted_cols[j] = {
                **info,
                "impute_value": mode_val,
                "mean": mean_val,
                "std": std_val,
            }

        elif ctype == "continuous":
            if info["needs_cleaning"]:
                col = _apply_continuous_sentinel_rules(col, info["sentinel_rules"])
            valid = col[~np.isnan(col)]
            median_val = float(np.median(valid)) if valid.size > 0 else 0.0
            imputed = np.where(np.isnan(col), median_val, col)
            mean_val = float(np.mean(imputed))
            std_val = float(np.std(imputed))
            if std_val == 0.0:
                std_val = 1.0
            fitted_cols[j] = {
                **info,
                "impute_value": median_val,
                "mean": mean_val,
                "std": std_val,
            }

    return {
        "fitted_cols": fitted_cols,
        "summary": summary,
        "one_hot_categorical": one_hot_categorical,
        "add_bias": add_bias,
    }


def transform_preprocessor(x, params):
    """Transform a dataset (train, validation, or test) using parameters fitted on x_train.

    Args:
        x: 2D numpy array of shape (N, D)
        params: dict returned by fit_preprocessor(x_train)

    Returns:
        x_processed: 2D numpy array of shape (N, D_out) with zero NaNs and standardized features.
    """
    n_samples = x.shape[0]
    fitted_cols = params["fitted_cols"]
    one_hot = params["one_hot_categorical"]

    feature_blocks = []
    if params["add_bias"]:
        feature_blocks.append(np.ones((n_samples, 1), dtype=float))

    for j in sorted(fitted_cols.keys()):
        info = fitted_cols[j]
        col = x[:, j].astype(float, copy=True)
        ctype = info["type"]

        if ctype == "binary":
            for nc in info["nan_codes"]:
                col[col == nc] = np.nan
            mapped = np.full(n_samples, info["impute_value"], dtype=float)
            mapped[col == info["val_zero"]] = 0.0
            mapped[col == info["val_one"]] = 1.0
            feature_blocks.append(mapped[:, None])

        elif ctype == "categorical":
            for nc in info["nan_codes"]:
                col[col == nc] = np.nan
            for zc in info["zero_codes"]:
                col[col == zc] = 0.0
            col = np.where(np.isnan(col), info["impute_value"], col)
            if one_hot:
                # One-hot encode (skip first category to avoid collinearity with bias column)
                cats = info["categories"][1:]
                if cats.size > 0:
                    oh = (col[:, None] == cats[None, :]).astype(float)
                    feature_blocks.append(oh)
            else:
                col_std = (col - info["mean"]) / info["std"]
                feature_blocks.append(col_std[:, None])

        elif ctype == "continuous":
            if info["needs_cleaning"]:
                col = _apply_continuous_sentinel_rules(col, info["sentinel_rules"])
            col = np.where(np.isnan(col), info["impute_value"], col)
            col_std = (col - info["mean"]) / info["std"]
            feature_blocks.append(col_std[:, None])

    return np.hstack(feature_blocks)


def fit_transform_preprocessor(x_train, **kwargs):
    """Convenience function that fits on x_train and returns (x_train_clean, params)."""
    params = fit_preprocessor(x_train, **kwargs)
    return transform_preprocessor(x_train, params), params
