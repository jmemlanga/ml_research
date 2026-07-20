
#Import necessary libraries

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate


#Settings that stay fixed while comparing hyperparameters

HERE = Path(__file__).parent #fix from Ai

FEATURES_CSV = HERE / "data" / "generated_features.csv"
RESULTS_CSV = HERE / "svr_results.csv"

N_SPLITS = 5
RANDOM_STATE = 0

CALCULATE_TEST_METRICS = True


#Loading the data and honouring the split

data = pd.read_csv(FEATURES_CSV)

train = data[data["split"] == "train"].copy()
test  = data[data["split"] == "test"].copy()


#Removing the columns that are not features

non_features = {"file_idx","smiles","split","dft_barrier"}

feature_cols = [c for c in data.columns if c not in non_features]

X_train = train[feature_cols]
y_train = train["dft_barrier"]

X_test = test[feature_cols]
y_test = test["dft_barrier"]


#The folds are built once so every configuration is scored on the same splits

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def svr_pipeline(C, epsilon, gamma):

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
    ])

    results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=kfold,
        scoring={"mae":"neg_mean_absolute_error",
                 "rmse":"neg_root_mean_squared_error",
                 "r_squared": "r2"}
    )

    fold_mae = -results["test_mae"]
    fold_rmse = -results["test_rmse"]
    fold_r2 = results["test_r_squared"]

    return {
        "C": C,
        "epsilon": epsilon,
        "gamma": gamma,
        "mae": np.mean(fold_mae),
        "std_mae": np.std(fold_mae, ddof=1),
        "rmse": np.mean(fold_rmse),
        "std_rmse": np.std(fold_rmse, ddof=1),
        "r2": np.mean(fold_r2),
        "std_r2": np.std(fold_r2, ddof=1),
    }


def svr_test(C, epsilon, gamma):

    #Only run once the hyperparameters are final

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "C": C,
        "epsilon": epsilon,
        "gamma": gamma,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }


def save_result(result, label):

    #Adds one row to the results file so runs build up instead of being lost

    row = pd.DataFrame([{"evaluation": label, **result}])
    row.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)


def main():

    result = svr_pipeline(C=500, epsilon=0.3, gamma=0.0007)
    save_result(result, "cv")

    print(f"C = {result['C']}, epsilon = {result['epsilon']}, gamma = {result['gamma']}")
    print(f"MAE  = {result['mae']:.4f} +/- {result['std_mae']:.4f}")
    print(f"RMSE = {result['rmse']:.4f} +/- {result['std_rmse']:.4f}")
    print(f"r2   = {result['r2']:.4f} +/- {result['std_r2']:.4f}")

    if CALCULATE_TEST_METRICS:
        test_result = svr_test(C=500, epsilon=0.3, gamma=0.0007)
        save_result(test_result, "test")
        print()
        print("Withheld test set")
        print(f"MAE  = {test_result['mae']:.4f}")
        print(f"RMSE = {test_result['rmse']:.4f}")
        print(f"r2   = {test_result['r2']:.4f}")
    else:
        print()
        print("Test evaluation is off. Set CALCULATE_TEST_METRICS = True when the hyperparameters are final.")


if __name__ == "__main__":
    main()

    