# Visual 1: Compare Test VE and Test MSE across RAW / STANDARDIZED / LOG1P
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 绝对路径（根据你的环境） ===
OUT_DIR = r"C:\Users\Grace\Desktop\FL25\CSE 5104\Project\Proj01\partB_outputs"
RAW_PERF_CSV = r"C:\Users\Grace\Desktop\FL25\CSE 5104\Project\Proj01\partB_outputs\raw_performance.csv"
STD_PERF_CSV = r"C:\Users\Grace\Desktop\FL25\CSE 5104\Project\Proj01\partB_outputs\std_performance.csv"
LOG_PERF_CSV = r"C:\Users\Grace\Desktop\FL25\CSE 5104\Project\Proj01\partB_outputs\log1p_performance.csv"

PVALS_RAW_CSV = r"C:\Users\Grace\Desktop\FL25\CSE 5104\Project\Proj01\partB_outputs\pvalues_raw.csv"
PVALS_LOG_CSV = r"C:\Users\Grace\Desktop\FL25\CSE 5104\Project\Proj01\partB_outputs\pvalues_log1p.csv"

# --- Load performance CSVs (支持“长表 metric,value”与“一行多列”两种格式) ---
raw_df = pd.read_csv(RAW_PERF_CSV)
lower = [c.lower() for c in raw_df.columns]
if "metric" in lower:
    metric_col = raw_df.columns[lower.index("metric")]
    value_col = raw_df.columns[lower.index("value")] if "value" in lower else raw_df.columns[1]
    raw_perf = {str(row[metric_col]): float(row[value_col]) for _, row in raw_df.iterrows()}
else:
    raw_perf = {c: float(raw_df.iloc[0][c]) for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])}

std_df = pd.read_csv(STD_PERF_CSV)
lower = [c.lower() for c in std_df.columns]
if "metric" in lower:
    metric_col = std_df.columns[lower.index("metric")]
    value_col = std_df.columns[lower.index("value")] if "value" in lower else std_df.columns[1]
    std_perf = {str(row[metric_col]): float(row[value_col]) for _, row in std_df.iterrows()}
else:
    std_perf = {c: float(std_df.iloc[0][c]) for c in std_df.columns if pd.api.types.is_numeric_dtype(std_df[c])}

log_df = pd.read_csv(LOG_PERF_CSV)
lower = [c.lower() for c in log_df.columns]
if "metric" in lower:
    metric_col = log_df.columns[lower.index("metric")]
    value_col = log_df.columns[lower.index("value")] if "value" in lower else log_df.columns[1]
    log_perf = {str(row[metric_col]): float(row[value_col]) for _, row in log_df.iterrows()}
else:
    log_perf = {c: float(log_df.iloc[0][c]) for c in log_df.columns if pd.api.types.is_numeric_dtype(log_df[c])}

# --- Build a compact dataframe for plotting ---
rows = []
for name, perf in [("RAW", raw_perf), ("STANDARDIZED", std_perf), ("LOG1P", log_perf)]:
    rows.append({
        "Setting": name,
        "Test_VE": perf.get("test_VE", perf.get("test_ve")),
        "Test_MSE": perf.get("test_MSE", perf.get("test_mse")),
    })
dfp = pd.DataFrame(rows)

# --- Plot Test VE ---
plt.figure()
plt.bar(dfp["Setting"], dfp["Test_VE"])
plt.title("Test VE by Feature Engineering Setting")
plt.xlabel("Setting")
plt.ylabel("Test VE")
plt.tight_layout()
plt.savefig(OUT_DIR + r"\visual_test_VE_by_setting.png", dpi=150)

# --- Plot Test MSE ---
plt.figure()
plt.bar(dfp["Setting"], dfp["Test_MSE"])
plt.title("Test MSE by Feature Engineering Setting")
plt.xlabel("Setting")
plt.ylabel("Test MSE")
plt.tight_layout()
plt.savefig(OUT_DIR + r"\visual_test_MSE_by_setting.png", dpi=150)

print("Saved:",
      OUT_DIR + r"\visual_test_VE_by_setting.png",
      "and",
      OUT_DIR + r"\visual_test_MSE_by_setting.png")

# ------------------------------------------------------------
# Visual 2: Significance comparison via -log10(p) for RAW vs LOG1P
# ------------------------------------------------------------

def load_pvals_abs(path):
    df = pd.read_csv(path)
    # Try to find columns
    term_col = None
    p_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("term", "variable", "name"):
            term_col = c
        if cl in ("p_value", "pvalue", "p-values", "p"):
            p_col = c
    if term_col is None:
        term_col = df.columns[0]
    if p_col is None:
        p_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    out = df[[term_col, p_col]].copy()
    out.columns = ["term", "p"]
    out["p"] = pd.to_numeric(out["p"], errors="coerce")
    return out

p_raw = load_pvals_abs(PVALS_RAW_CSV)
p_log = load_pvals_abs(PVALS_LOG_CSV)

# align terms
terms = sorted(set(p_raw["term"]).intersection(set(p_log["term"])))
dfc = pd.DataFrame({"term": terms})
dfc["p_raw"] = dfc["term"].map(dict(zip(p_raw["term"], p_raw["p"])))
dfc["p_log"] = dfc["term"].map(dict(zip(p_log["term"], p_log["p"])))
dfc["mlog10_raw"] = -np.log10(dfc["p_raw"].clip(lower=1e-300))
dfc["mlog10_log"] = -np.log10(dfc["p_log"].clip(lower=1e-300))

x = np.arange(len(terms))
width = 0.4

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, dfc["mlog10_raw"], width=width, label="RAW")
plt.bar(x + width/2, dfc["mlog10_log"], width=width, label="LOG1P")
plt.axhline(-np.log10(0.05), linestyle="--")  # significance threshold line
plt.xticks(x, terms, rotation=30, ha="right")
plt.title("Significance Shift by Feature Engineering: -log10(p)")
plt.xlabel("Term")
plt.ylabel("-log10(p)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + r"\visual_significance_shift.png", dpi=150)

print("Saved:", OUT_DIR + r"\visual_significance_shift.png")
