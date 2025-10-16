# project1_partB_ols_analysis.py
# OLS regression analysis for RAW / STANDARDIZED / LOG1P predictors.
# Saves performance CSVs and p-value CSVs.


import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

CSV_PATH = "Concrete_Data.csv"

def load_split(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    y_col = [c for c in df.columns if "compressive" in c.lower()][0]
    X_cols = [c for c in df.columns if c != y_col]
    test_idx = np.arange(500, 630)
    train_idx = np.array([i for i in range(len(df)) if i not in test_idx])
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    return df, train, test, X_cols, y_col

def mse(y, yhat): return float(np.mean((y - yhat)**2))
def VE(y, yhat):
    var = float(np.var(y, ddof=0))
    return float('nan') if var==0 else 1.0 - mse(y,yhat)/var

def fit_and_eval(Xtr, Xte, ytr, yte):
    res = sm.OLS(ytr, sm.add_constant(Xtr, prepend=True)).fit()
    yhat_tr = res.predict(sm.add_constant(Xtr, prepend=True))
    yhat_te = res.predict(sm.add_constant(Xte, prepend=True))
    return res, dict(train_MSE=mse(ytr,yhat_tr), train_VE=VE(ytr,yhat_tr),
                     test_MSE=mse(yte,yhat_te), test_VE=VE(yte,yhat_te),
                     rsq=res.rsquared, rsq_adj=res.rsquared_adj)

def main():
    df, train, test, X_cols, y_col = load_split(CSV_PATH)
    Xtr_raw, ytr = train[X_cols].to_numpy(float), train[y_col].to_numpy(float)
    Xte_raw, yte = test[X_cols].to_numpy(float),  test[y_col].to_numpy(float)
    outdir="partB_outputs"; os.makedirs(outdir, exist_ok=True)

    # RAW
    res_raw, perf_raw = fit_and_eval(Xtr_raw, Xte_raw, ytr, yte)
    pd.DataFrame({"metric":["train_MSE","train_VE","test_MSE","test_VE","train_R2","train_R2_adj"],
                  "value":[perf_raw["train_MSE"],perf_raw["train_VE"],perf_raw["test_MSE"],perf_raw["test_VE"],perf_raw["rsq"],perf_raw["rsq_adj"]]}).to_csv(os.path.join(outdir,"raw_performance.csv"), index=False)
    pd.Series(res_raw.pvalues, index=["const"]+X_cols).reset_index().rename(columns={"index":"Term",0:"p_value"}).to_csv(os.path.join(outdir,"pvalues_raw.csv"), index=False)

    # STANDARDIZED
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0, ddof=0); sd[sd==0]=1.0
    Xtr_std, Xte_std = (Xtr_raw-mu)/sd, (Xte_raw-mu)/sd
    res_std, perf_std = fit_and_eval(Xtr_std, Xte_std, ytr, yte)
    pd.DataFrame({"metric":["train_MSE","train_VE","test_MSE","test_VE","train_R2","train_R2_adj"],
                  "value":[perf_std["train_MSE"],perf_std["train_VE"],perf_std["test_MSE"],perf_std["test_VE"],perf_std["rsq"],perf_std["rsq_adj"]]}).to_csv(os.path.join(outdir,"std_performance.csv"), index=False)
    pd.Series(res_std.pvalues, index=["const"]+X_cols).reset_index().rename(columns={"index":"Term",0:"p_value"}).to_csv(os.path.join(outdir,"pvalues_standardized.csv"), index=False)

    # LOG1P
    Xtr_log, Xte_log = np.log1p(Xtr_raw), np.log1p(Xte_raw)
    res_log, perf_log = fit_and_eval(Xtr_log, Xte_log, ytr, yte)
    pd.DataFrame({"metric":["train_MSE","train_VE","test_MSE","test_VE","train_R2","train_R2_adj"],
                  "value":[perf_log["train_MSE"],perf_log["train_VE"],perf_log["test_MSE"],perf_log["test_VE"],perf_log["rsq"],perf_log["rsq_adj"]]}).to_csv(os.path.join(outdir,"log1p_performance.csv"), index=False)
    pd.Series(res_log.pvalues, index=["const"]+X_cols).reset_index().rename(columns={"index":"Term",0:"p_value"}).to_csv(os.path.join(outdir,"pvalues_log1p.csv"), index=False)

    print("Part B outputs to", outdir)

if __name__ == "__main__":
    main()
