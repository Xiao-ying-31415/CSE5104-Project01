# project1_partA.py
# Gradient Descent for Linear Regression (Univariate + Multivariate)
# Train/Test split: rows 501-630 (1-based) are test

import numpy as np
import pandas as pd
import os
from typing import Tuple

CSV_PATH = "Concrete_Data.csv"

def load_split(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    y_col = [c for c in df.columns if "compressive" in c.lower()][0]
    X_cols = [c for c in df.columns if c != y_col]
    n = len(df)
    assert n == 1030, f"Expected 1030 rows, found {n}."
    test_idx = np.arange(500, 630)
    train_idx = np.array([i for i in range(n) if i not in test_idx])
    train = df.iloc[train_idx].reset_index(drop=True)
    test  = df.iloc[test_idx].reset_index(drop=True)
    return train, test, X_cols, y_col

def mse(y, yhat) -> float:
    return float(np.mean((y - yhat) ** 2))

def variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """VE = 1 - MSE/Var(y_true) using population variance (ddof=0)."""
    var = float(np.var(y_true, ddof=0))
    return float('nan') if var == 0.0 else 1.0 - mse(y_true, y_pred) / var

def standardize_fit_transform(X: np.ndarray):
    """Z-score on training data; return (Z, mu, sd)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd, mu, sd

def standardize_transform(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    sd_safe = sd.copy()
    sd_safe[sd_safe == 0.0] = 1.0
    return (X - mu) / sd_safe

# ---------------------
# Gradient Descent
# ---------------------
def gd_univariate(x: np.ndarray, y: np.ndarray, alpha=0.01, iters=60000, m0=0.0, b0=0.0):
    """Univariate linear regression via batch GD: y ≈ m*x + b."""
    n = len(x)
    m = float(m0)
    b = float(b0)
    for _ in range(iters):
        yhat = m * x + b
        err = yhat - y
        grad_m = (2.0 / n) * np.dot(x, err)
        grad_b = (2.0 / n) * np.sum(err)
        m -= alpha * grad_m
        b -= alpha * grad_b
    return m, b

def gd_multivariate(X: np.ndarray, y: np.ndarray, alpha=0.01, iters=60000, w0=None, b0=0.0):
    """Multivariate linear regression via batch GD: y ≈ Xw + b."""
    n, p = X.shape
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=float).copy()
    b = float(b0)
    for _ in range(iters):
        yhat = X @ w + b
        err = yhat - y
        grad_w = (2.0 / n) * (X.T @ err)
        grad_b = (2.0 / n) * np.sum(err)
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b

# ---------------------
# Part A runners
# ---------------------
def run_set1_univariate(train: pd.DataFrame, test: pd.DataFrame, X_cols, y_col, outdir: str):
    """Q1.1: Standardized predictors, raw y, univariate GD per feature."""
    Xtr_raw = train[X_cols].to_numpy(float)
    Xte_raw = test[X_cols].to_numpy(float)
    ytr = train[y_col].to_numpy(float)
    yte = test[y_col].to_numpy(float)

    Ztr, mu, sd = standardize_fit_transform(Xtr_raw)
    Zte = standardize_transform(Xte_raw, mu, sd)

    rows = []
    for j, name in enumerate(X_cols):
        m, b = gd_univariate(Ztr[:, j], ytr, alpha=0.01, iters=60000)
        yhat_tr = m * Ztr[:, j] + b
        yhat_te = m * Zte[:, j] + b
        rows.append(dict(
            Predictor=name, m=m, b=b,
            Train_MSE=mse(ytr, yhat_tr), Train_VE=variance_explained(ytr, yhat_tr),
            Test_MSE=mse(yte, yhat_te), Test_VE=variance_explained(yte, yhat_te)
        ))
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "Q1_1_Set1_univariate_results.csv"), index=False)

def run_set2_univariate(train: pd.DataFrame, test: pd.DataFrame, X_cols, y_col, outdir: str):
    """Q1.2: Raw predictors, raw y, univariate GD per feature."""
    Xtr_raw = train[X_cols].to_numpy(float)
    Xte_raw = test[X_cols].to_numpy(float)
    ytr = train[y_col].to_numpy(float)
    yte = test[y_col].to_numpy(float)

    rows = []
    for j, name in enumerate(X_cols):
        m, b = gd_univariate(Xtr_raw[:, j], ytr, alpha=2e-8, iters=60000)
        yhat_tr = m * Xtr_raw[:, j] + b
        yhat_te = m * Xte_raw[:, j] + b
        rows.append(dict(
            Predictor=name, m=m, b=b,
            Train_MSE=mse(ytr, yhat_tr), Train_VE=variance_explained(ytr, yhat_tr),
            Test_MSE=mse(yte, yhat_te), Test_VE=variance_explained(yte, yhat_te)
        ))
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "Q1_2_Set2_univariate_results.csv"), index=False)

def run_set1_multivariate(train: pd.DataFrame, test: pd.DataFrame, X_cols, y_col, outdir: str):
    """Q2.3: Standardized predictors, multivariate GD."""
    Xtr_raw = train[X_cols].to_numpy(float)
    Xte_raw = test[X_cols].to_numpy(float)
    ytr = train[y_col].to_numpy(float)
    yte = test[y_col].to_numpy(float)

    Ztr, mu, sd = standardize_fit_transform(Xtr_raw)
    Zte = standardize_transform(Xte_raw, mu, sd)

    w, b = gd_multivariate(Ztr, ytr, alpha=0.01, iters=60000)
    yhat_tr = Ztr @ w + b
    yhat_te = Zte @ w + b

    pd.DataFrame({"Predictor": X_cols, "w": w}).to_csv(os.path.join(outdir, "Q2_3_Set1_multivariate_weights.csv"), index=False)
    pd.DataFrame([dict(
        alpha=0.01, b=b,
        train_MSE=mse(ytr, yhat_tr), train_VE=variance_explained(ytr, yhat_tr),
        test_MSE=mse(yte, yhat_te), test_VE=variance_explained(yte, yhat_te)
    )]).to_csv(os.path.join(outdir, "Q2_3_Set1_multivariate_summary.csv"), index=False)

def run_set2_multivariate(train: pd.DataFrame, test: pd.DataFrame, X_cols, y_col, outdir: str):
    """Q2.4: Raw predictors, multivariate GD."""
    Xtr_raw = train[X_cols].to_numpy(float)
    Xte_raw = test[X_cols].to_numpy(float)
    ytr = train[y_col].to_numpy(float)
    yte = test[y_col].to_numpy(float)

    w, b = gd_multivariate(Xtr_raw, ytr, alpha=2e-8, iters=60000)
    yhat_tr = Xtr_raw @ w + b
    yhat_te = Xte_raw @ w + b

    pd.DataFrame({"Predictor": X_cols, "w": w}).to_csv(os.path.join(outdir, "Q2_4_Set2_multivariate_weights.csv"), index=False)
    pd.DataFrame([dict(
        alpha=2e-8, b=b,
        train_MSE=mse(ytr, yhat_tr), train_VE=variance_explained(ytr, yhat_tr),
        test_MSE=mse(yte, yhat_te), test_VE=variance_explained(yte, yhat_te)
    )]).to_csv(os.path.join(outdir, "Q2_4_Set2_multivariate_summary.csv"), index=False)

# ---------------------
# Q2.1 and Q2.2: one-step updates
# ---------------------
def gd_one_step(X, y, alpha=0.1, w0=None, b0=1.0):
    """Do exactly ONE batch-GD step with factor 2/n (same as gd_multivariate)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = X.shape
    w = np.ones(p, dtype=float) if (w0 is None) else np.asarray(w0, dtype=float).copy()
    b = float(b0)
    yhat = X @ w + b
    err = yhat - y
    grad_w = (2.0 / n) * (X.T @ err)
    grad_b = (2.0 / n) * np.sum(err)
    w_new = w - alpha * grad_w
    b_new = b - alpha * grad_b
    return w_new, b_new

def run_q2_1():
    """Q2.1: Single sample, alpha=0.1, w0=[1,1,1], b0=1; one step."""
    w, b = gd_one_step(X=[[3, 4, 5]], y=[4], alpha=0.1, w0=[1,1,1], b0=1.0)
    print("Q2.1 one-step result (w, b):", w, b)

def run_q2_2():
    """Q2.2: Multiple sample, alpha=0.1, w0=[1,1,1], b0=1; one step."""
    X_5 = [
        [3, 4, 4],
        [4, 2, 1],
        [10, 2, 5],
        [3, 4, 5],
        [11, 1, 1],
    ]
    y_5 = [3, 2, 8, 4, 5]
    if len(X_5) == 5 and len(y_5) == 5:
        w, b = gd_one_step(X_5, y_5, alpha=0.1, w0=[1,1,1], b0=1.0)
        print("Q2.2 one-step result (w, b):", w, b)
    else:
        print("Q2.2 template: please paste 5 rows into X_5 and y_5, then run again.")

# ---------------------
# Main
# ---------------------
def main():
    outdir = "partA_outputs"
    os.makedirs(outdir, exist_ok=True)

    # Load split
    train, test, X_cols, y_col = load_split(CSV_PATH)

    # Part A
    run_set1_univariate(train, test, X_cols, y_col, outdir)   # Q1.1
    run_set2_univariate(train, test, X_cols, y_col, outdir)   # Q1.2
    run_set1_multivariate(train, test, X_cols, y_col, outdir) # Q2.3
    run_set2_multivariate(train, test, X_cols, y_col, outdir) # Q2.4
    print("Part A outputs written to", outdir)

    # Q2
    run_q2_1()   # single-sample one-step
    run_q2_2()

if __name__ == '__main__':
    main()
