#!/usr/bin/env python3
"""Train and evaluate an RBF support vector regression model."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# =============================================================================
# SETTINGS TO EDIT
# =============================================================================

# Folder containing the file produced by generate_features.py.


FEATURES_DIR = Path("data")
FEATURES_CSV = FEATURES_DIR / "generated_features.csv"

# The results from each run are written here.
OUTPUT_DIR = Path(".")
RESULTS_CSV = OUTPUT_DIR / "svr_results.csv"

# SVR hyperparameters to experiment with.
C = 5.0
EPSILON = 0.1
GAMMA = 0.01

# Keep these fixed while comparing hyperparameters.
# Folds are the name for the number of splits in cross-validation. Random state is used to shuffle the training data before splitting into folds.
N_SPLITS = 5
RANDOM_STATE = 0



# Keep this False during hyperparameter optimisation.
# Change it to True only after choosing the final hyperparameters from CV.
CALCULATE_TEST_METRICS = False

# Columns that describe each molecule but are not model features.
NON_FEATURE_COLUMNS = {
    "file_idx",
    "smiles",
    "dft_barrier",
    "split",
}


def make_model() -> Pipeline:
    """Create the preprocessing steps and the RBF-SVR model."""
    return Pipeline(
        steps=[
            # Replace missing descriptor values using the training-set median.
            ("imputer", SimpleImputer(strategy="median")),
            # Remove descriptor columns that are constant in the training data.
            ("variance", VarianceThreshold()),
            # SVR works best when descriptor columns have similar scales.
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=C,
                    epsilon=EPSILON,
                    gamma=GAMMA,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for one set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R2 is undefined when a split contains fewer than two samples.
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def add_hyperparameters(result: dict[str, object]) -> dict[str, object]:
    """Add the current hyperparameters to a results row."""
    return {
        **result,
        "kernel": "rbf",
        "C": C,
        "epsilon": EPSILON,
        "gamma": GAMMA,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "calculate_test_metrics": CALCULATE_TEST_METRICS,
    }

def main() -> None:
    features_path = FEATURES_CSV.expanduser().resolve()
    results_path = RESULTS_CSV.expanduser().resolve()

    if not features_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)

    required_columns = {"dft_barrier", "split"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Feature file is missing required column(s): {sorted(missing_columns)}"
        )

    data["split"] = data["split"].astype(str).str.strip().str.lower()
    data["dft_barrier"] = pd.to_numeric(data["dft_barrier"], errors="raise")

    unexpected_splits = sorted(set(data["split"]) - {"train", "test"})
    if unexpected_splits:
        raise ValueError(
            "Only train/test splits are expected; found: "
            f"{unexpected_splits}"
        )
    if data["dft_barrier"].isna().any():
        raise ValueError("The dft_barrier column contains missing values")

    train_data = data.loc[data["split"] == "train"].copy()
    test_data = data.loc[data["split"] == "test"].copy()

    if len(train_data) < N_SPLITS:
        raise ValueError(
            f"{N_SPLITS}-fold cross-validation requires at least "
            f"{N_SPLITS} training rows; found {len(train_data)}"
        )
    if CALCULATE_TEST_METRICS and test_data.empty:
        raise ValueError(
            "Test metrics are enabled, but no rows with split='test' were found"
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No descriptor columns were found")

    X_train = train_data[feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    y_train = train_data["dft_barrier"].to_numpy(dtype=float)

    # Do not prepare or inspect the test-set values during hyperparameter tuning.
    if CALCULATE_TEST_METRICS:
        X_test = test_data[feature_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        y_test = test_data["dft_barrier"].to_numpy(dtype=float)

    print("SVR hyperparameters")
    print(f"  kernel  = rbf")
    print(f"  C       = {C}")
    print(f"  epsilon = {EPSILON}")
    print(f"  gamma   = {GAMMA}")
    print()
    print(f"Training molecules: {len(train_data)}")
    print(f"Test molecules:     {len(test_data)}")
    print(f"Descriptor columns: {len(feature_columns)}")
    print()

    # The pipeline is fitted separately inside every fold. This prevents
    # information from a validation fold leaking into preprocessing.
    cross_validator = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = list(cross_validator.split(X_train))

    cv_scores = cross_validate(
        estimator=make_model(),
        X=X_train,
        y=y_train,
        cv=folds,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        error_score="raise",
    )

    fold_mae = -cv_scores["test_mae"]
    fold_rmse = -cv_scores["test_rmse"]
    fold_r2 = cv_scores["test_r2"]

    result_rows = []
    for fold_number, (_, validation_indices) in enumerate(folds, start=1):
        result_rows.append(
            add_hyperparameters(
                {
                    "evaluation": f"cv_fold_{fold_number}",
                    "n_samples": len(validation_indices),
                    "mae": fold_mae[fold_number - 1],
                    "rmse": fold_rmse[fold_number - 1],
                    "r2": fold_r2[fold_number - 1],
                }
            )
        )

    result_rows.append(
        add_hyperparameters(
            {
                "evaluation": "cv_mean",
                "n_samples": len(train_data),
                "mae": float(np.mean(fold_mae)),
                "rmse": float(np.mean(fold_rmse)),
                "r2": float(np.mean(fold_r2)),
            }
        )
    )
    result_rows.append(
        add_hyperparameters(
            {
                "evaluation": "cv_standard_deviation",
                "n_samples": len(train_data),
                "mae": float(np.std(fold_mae, ddof=1)),
                "rmse": float(np.std(fold_rmse, ddof=1)),
                "r2": float(np.std(fold_r2, ddof=1)),
            }
        )
    )

    # Only evaluate the test set after the final hyperparameters have been chosen.
    test_metrics = None

    if CALCULATE_TEST_METRICS:
        final_model = make_model()
        final_model.fit(X_train, y_train)

        test_predictions = final_model.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_predictions)

        result_rows.append(
            add_hyperparameters(
                {
                    "evaluation": "withheld_test",
                    "n_samples": len(test_data),
                    **test_metrics,
                }
            )
        )

    results = pd.DataFrame(result_rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    cv_table = results.loc[
        results["evaluation"].str.startswith("cv_fold_"),
        ["evaluation", "n_samples", "mae", "rmse", "r2"],
    ]

    print("5-fold cross-validation results")
    print(cv_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(
        "CV mean: "
        f"MAE = {np.mean(fold_mae):.4f}, "
        f"RMSE = {np.mean(fold_rmse):.4f}, "
        f"R2 = {np.mean(fold_r2):.4f}"
    )
    print(
        "CV standard deviation: "
        f"MAE = {np.std(fold_mae, ddof=1):.4f}, "
        f"RMSE = {np.std(fold_rmse, ddof=1):.4f}, "
        f"R2 = {np.std(fold_r2, ddof=1):.4f}"
    )
    print()

    if CALCULATE_TEST_METRICS:
        print("Withheld test-set results")
        print(f"  MAE  = {test_metrics['mae']:.4f}")
        print(f"  RMSE = {test_metrics['rmse']:.4f}")
        print(f"  R2   = {test_metrics['r2']:.4f}")
    else:
        print("Withheld test-set evaluation is disabled.")
        print(
            "Select the final hyperparameters using cross-validation, "
            "then set CALCULATE_TEST_METRICS = True."
        )

    print()
    print(f"Saved all results to: {results_path}")


if __name__ == "__main__":
    main()
