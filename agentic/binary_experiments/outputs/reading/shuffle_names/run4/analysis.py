import pandas as pd
from statsmodels.stats.weightstats import ttest_ind

# Load data
_df = pd.read_csv("reading.csv")

# Column mapping based on value inspection in this shuffled dataset
# - language: binary indicator of Reader View (0/1)
# - device: dyslexia status (0 = no dyslexia, 1 = dyslexia, 2 = severe dyslexia)
# - running_time: reading speed (higher is faster; aligns with words/time)

_df = _df.copy()
_df["reader_view"] = _df["language"]
_df["dyslexia_status"] = _df["device"]
_df["reading_speed"] = _df["running_time"]

# Focus on individuals with dyslexia (1 or 2)
mask = _df["dyslexia_status"].isin([1.0, 2.0])
sub = _df.loc[mask & _df["reader_view"].notna() & _df["reading_speed"].notna()].copy()

rv_on = sub.loc[sub["reader_view"] == 1, "reading_speed"]
rv_off = sub.loc[sub["reader_view"] == 0, "reading_speed"]

# Summary statistics
summary = {
    "n_reader_view_on": int(rv_on.shape[0]),
    "n_reader_view_off": int(rv_off.shape[0]),
    "mean_speed_on": float(rv_on.mean()),
    "mean_speed_off": float(rv_off.mean()),
    "median_speed_on": float(rv_on.median()),
    "median_speed_off": float(rv_off.median()),
    "mean_diff_on_minus_off": float(rv_on.mean() - rv_off.mean()),
}

# Welch's t-test
if len(rv_on) > 1 and len(rv_off) > 1:
    t_stat, p_value, df_dof = ttest_ind(rv_on, rv_off, usevar="unequal")
    summary["t_stat"] = float(t_stat)
    summary["p_value"] = float(p_value)
    summary["df"] = float(df_dof)

print("Dyslexia subset summary (reading_speed ~ Reader View):")
for k, v in summary.items():
    print(f"{k}: {v}")
