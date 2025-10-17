import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def mse(y, yhat): return float(np.mean((y - yhat) ** 2))
def variance_explained(y_true, y_pred):
    var = float(np.var(y_true, ddof=0))
    return float('nan') if var==0.0 else 1.0 - mse(y_true, y_pred) / var

def save_loss_plot(loss, title, outpath):
    plt.figure(); plt.plot(loss); plt.title(title)
    plt.xlabel("Iteration"); plt.ylabel("MSE"); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def standardize_fit_transform(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd==0.0] = 1.0
    return (X - mu) / sd, mu, sd

def standardize_transform(X, mu, sd):
    sd = sd.copy(); sd[sd==0.0] = 1.0
    return (X - mu) / sd

def gd_univariate(x, y, alpha=0.01, iters=60000, m0=0.0, b0=0.0, record_loss=False):
    n = len(x); m = float(m0); b = float(b0); loss_hist = []
    for _ in range(iters):
        yhat = m * x + b; err = yhat - y
        if record_loss: loss_hist.append(mse(y, yhat))
        grad_m = (2.0/n) * np.dot(x, err); grad_b = (2.0/n) * np.sum(err)
        m -= alpha * grad_m; b -= alpha * grad_b
    return (m, b, loss_hist) if record_loss else (m, b)

def gd_multivariate(X, y, alpha=0.01, iters=60000, w0=None, b0=0.0, record_loss=False):
    n, p = X.shape; w = np.zeros(p) if w0 is None else np.array(w0, float).copy(); b = float(b0); loss_hist = []
    for _ in range(iters):
        yhat = X @ w + b; err = yhat - y
        if record_loss: loss_hist.append(mse(y, yhat))
        grad_w = (2.0/n) * (X.T @ err); grad_b = (2.0/n) * np.sum(err)
        w -= alpha * grad_w; b -= alpha * grad_b
    return (w, b, loss_hist) if record_loss else (w, b)

def load_split(csv_path):
    df = pd.read_csv(csv_path); df.columns = [c.strip() for c in df.columns]
    y_col = [c for c in df.columns if "compressive" in c.lower()][0]
    X_cols = [c for c in df.columns if c != y_col]
    n = len(df); assert n == 1030, f"Expected 1030 rows, found {n}."
    test_idx = np.arange(500, 630)  # 0-based indices for rows 501..630 (1-based)
    train_idx = np.array([i for i in range(n) if i not in test_idx])
    train = df.iloc[train_idx].reset_index(drop=True); test = df.iloc[test_idx].reset_index(drop=True)
    return train, test, X_cols, y_col

def main(csv_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test, X_cols, y_col = load_split(str(csv_path))
    Xtr_raw = train[X_cols].to_numpy(float); Xte_raw = test[X_cols].to_numpy(float)
    ytr = train[y_col].to_numpy(float); yte = test[y_col].to_numpy(float)

    rows = []

    # Univariate — Set1 (Standardized)
    Ztr, mu_all, sd_all = standardize_fit_transform(Xtr_raw); Zte = standardize_transform(Xte_raw, mu_all, sd_all)
    for j, name in enumerate(X_cols):
        m, b = gd_univariate(Ztr[:, j], ytr, alpha=0.01, iters=60000)
        yhat_tr = m * Ztr[:, j] + b; yhat_te = m * Zte[:, j] + b
        rows.append(dict(Model="Univariate", Preprocess="Standardized", Predictor=name, m=m, b=b,
                         Train_MSE=mse(ytr, yhat_tr), Train_R2_like=variance_explained(ytr, yhat_tr),
                         Test_MSE=mse(yte, yhat_te),  Test_R2_like=variance_explained(yte, yhat_te)))

    # Univariate — Set2 (Raw): optimize in z-space then map back
    alpha_grid = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]; iters_grid = [20000, 60000, 120000]
    for j, name in enumerate(X_cols):
        xtr = Xtr_raw[:, j]; xte = Xte_raw[:, j]
        mu = float(xtr.mean()); sd = float(xtr.std(ddof=0)) or 1.0
        ztr = (xtr - mu) / sd; zte = (xte - mu) / sd
        best = {"alpha": None, "iters": None, "m_raw": None, "b_raw": None,
                "Train_MSE": float("inf"), "Train_R2_like": -float("inf"),
                "Test_MSE": float("inf"),  "Test_R2_like": -float("inf")}
        b0 = float(ytr.mean())
        for iters in iters_grid:
            for alpha in alpha_grid:
                m_z, b_z = gd_univariate(ztr, ytr, alpha=alpha, iters=iters, m0=0.0, b0=b0)
                m_raw = m_z / sd; b_raw = b_z - (m_z * mu / sd)
                yhat_tr = m_raw * xtr + b_raw; yhat_te = m_raw * xte + b_raw
                tr_mse = mse(ytr, yhat_tr); tr_r2l = variance_explained(ytr, yhat_tr)
                te_mse = mse(yte, yhat_te); te_r2l = variance_explained(yte, yhat_te)
                if tr_r2l > best["Train_R2_like"]:
                    best.update(dict(alpha=alpha, iters=iters, m_raw=m_raw, b_raw=b_raw,
                                     Train_MSE=tr_mse, Train_R2_like=tr_r2l,
                                     Test_MSE=te_mse,  Test_R2_like=te_r2l))
        rows.append(dict(Model="Univariate", Preprocess="Raw", Predictor=name,
                         m=best["m_raw"], b=best["b_raw"], alpha=best["alpha"], iters=best["iters"],
                         Train_MSE=best["Train_MSE"], Train_R2_like=best["Train_R2_like"],
                         Test_MSE=best["Test_MSE"],  Test_R2_like=best["Test_R2_like"]))

    # Multivariate — Set1 (Standardized)
    w_std, b_std, loss_std = gd_multivariate(Ztr, ytr, alpha=0.01, iters=60000, record_loss=True)
    yhat_tr = Ztr @ w_std + b_std; yhat_te = Zte @ w_std + b_std
    rows.append(dict(Model="Multivariate", Preprocess="Standardized", Predictor="ALL",
                     Train_MSE=mse(ytr, yhat_tr), Train_R2_like=variance_explained(ytr, yhat_tr),
                     Test_MSE=mse(yte, yhat_te),  Test_R2_like=variance_explained(yte, yhat_te)))
    save_loss_plot(loss_std, "Multivariate (Standardized) — Training Loss", out_dir / "Q2_3_Set1_multivariate_loss.png")
    pd.DataFrame({"Predictor": X_cols, "w": w_std}).to_csv(out_dir / "Q2_3_Set1_multivariate_weights.csv", index=False)

    # Multivariate — Set2 (Raw)
    w_raw, b_raw, loss_raw = gd_multivariate(Xtr_raw, ytr, alpha=2e-8, iters=60000, record_loss=True)
    yhat_tr = Xtr_raw @ w_raw + b_raw; yhat_te = Xte_raw @ w_raw + b_raw
    rows.append(dict(Model="Multivariate", Preprocess="Raw", Predictor="ALL",
                     Train_MSE=mse(ytr, yhat_tr), Train_R2_like=variance_explained(ytr, yhat_tr),
                     Test_MSE=mse(yte, yhat_te),  Test_R2_like=variance_explained(yte, yhat_te)))
    save_loss_plot(loss_raw, "Multivariate (Raw) — Training Loss", out_dir / "Q2_4_Set2_multivariate_loss.png")
    pd.DataFrame({"Predictor": X_cols, "w": w_raw}).to_csv(out_dir / "Q2_4_Set2_multivariate_weights.csv", index=False)

    out_cols = ["Model","Preprocess","Predictor","m","b","alpha","iters","Train_MSE","Train_R2_like","Test_MSE","Test_R2_like"]
    df_all = pd.DataFrame(rows)[out_cols]
    df_all.to_csv(out_dir / "All_Results_Summary.csv", index=False)
    print("Saved results to", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Concrete_Data.csv",
                            help="Path to Concrete_Data.csv")
    parser.add_argument("--out", type=str, default="partA_results_outputs",
                            help="Output directory")
    args = parser.parse_args()

    main(Path(args.csv), Path(args.out))
