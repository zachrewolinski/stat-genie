import json
import os

# Ensure matplotlib cache uses a writable directory before any plotting libs import
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Focus on individuals with dyslexia
# dyslexia_bin == 1 indicates dyslexia (including severe)
df_dys = df[df["dyslexia_bin"] == 1].copy()

# Basic counts
counts = df_dys["reader_view"].value_counts().to_dict()

# Descriptive stats by reader_view
summary = df_dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Log-transform to reduce influence of extreme outliers
# Add 1 to handle any zeros (though speed min seems >0)
df_dys["log_speed"] = np.log1p(df_dys["speed"])
log_summary = df_dys.groupby("reader_view")["log_speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Welch t-test on raw speed and log_speed
raw_ttest = pg.ttest(df_dys.loc[df_dys["reader_view"] == 1, "speed"],
                     df_dys.loc[df_dys["reader_view"] == 0, "speed"],
                     correction=True)
log_ttest = pg.ttest(df_dys.loc[df_dys["reader_view"] == 1, "log_speed"],
                     df_dys.loc[df_dys["reader_view"] == 0, "log_speed"],
                     correction=True)

# Effect size (Cohen's d) from pingouin ttest
raw_d = float(raw_ttest["cohen-d"].iloc[0])
log_d = float(log_ttest["cohen-d"].iloc[0])

# Paired analysis for participants who experienced both reader_view conditions
pivot = df_dys.pivot_table(index="uuid", columns="reader_view", values="speed", aggfunc="mean")
paired = pivot.dropna()
paired_diff = paired[1] - paired[0]
paired_ttest = pg.ttest(paired[1], paired[0], paired=True)

# OLS with cluster-robust SE by participant, controlling for page
# Use log_speed to reduce skew
ols = smf.ols("log_speed ~ reader_view + C(page_id)", data=df_dys).fit(
    cov_type="cluster", cov_kwds={"groups": df_dys["uuid"]}
)

conf_int = {}
for idx, row in ols.conf_int().iterrows():
    conf_int[idx] = [float(row[0]), float(row[1])]

results = {
    "counts": counts,
    "summary": summary.to_dict(orient="records"),
    "log_summary": log_summary.to_dict(orient="records"),
    "raw_ttest": raw_ttest.to_dict(orient="records"),
    "log_ttest": log_ttest.to_dict(orient="records"),
    "raw_cohen_d": raw_d,
    "log_cohen_d": log_d,
    "paired_n": int(len(paired)),
    "paired_ttest": paired_ttest.to_dict(orient="records"),
    "paired_mean_diff": float(paired_diff.mean()) if len(paired) > 0 else None,
    "ols_params": ols.params.to_dict(),
    "ols_pvalues": ols.pvalues.to_dict(),
    "ols_conf_int": conf_int,
}

def to_jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    return obj

print(json.dumps(to_jsonable(results), indent=2))
