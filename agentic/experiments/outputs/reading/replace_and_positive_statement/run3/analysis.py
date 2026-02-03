import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Identify dyslexia group
# If dyslexia_bin is not binary, infer two clusters and treat the higher-mean cluster as dyslexia.
if "dyslexia_bin" in df.columns:
    if df["dyslexia_bin"].dropna().nunique() <= 3:
        dys_indicator = df["dyslexia_bin"]
    else:
        vals = df[["dyslexia_bin"]].dropna()
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vals)
        cluster_means = {
            c: vals.iloc[clusters == c]["dyslexia_bin"].mean() for c in [0, 1]
        }
        high_cluster = max(cluster_means, key=cluster_means.get)
        dys_indicator = pd.Series(clusters == high_cluster, index=vals.index)
        df.loc[vals.index, "_dyslexia_group"] = dys_indicator.astype(int)
        dys_indicator = df["_dyslexia_group"].fillna(0)
else:
    dys_indicator = (df.get("dyslexia", 0) > 0).astype(int)

df_dys = df[dys_indicator == 1].copy()

# Remove non-positive speeds for log
# (speed should be positive, but guard just in case)
df_dys = df_dys[df_dys["speed"] > 0].copy()

# Descriptive stats
means = df_dys.groupby("reader_view")["speed"].agg(["mean", "median", "count"]).reset_index()

# Paired analysis: per-uuid mean speed in each condition
pivot = (
    df_dys.groupby(["uuid", "reader_view"])["speed"]
    .mean()
    .unstack("reader_view")
)
paired = pivot.dropna()  # keep uuids with both conditions

paired_result = None
if len(paired) >= 5:
    diffs = paired[1] - paired[0]
    paired_result = {
        "n_pairs": int(len(paired)),
        "mean_reader_view": float(paired[1].mean()),
        "mean_no_reader_view": float(paired[0].mean()),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(diffs.median()),
        "share_faster_in_reader_view": float((diffs > 0).mean()),
    }

# Regression models
# Model 1: participant fixed effects (within-subject), minimal controls
fe_terms = ["reader_view"]
if "page_id" in df_dys.columns and df_dys["page_id"].dropna().nunique() > 1:
    fe_terms.append("C(page_id)")
if "uuid" in df_dys.columns and df_dys["uuid"].dropna().nunique() > 1:
    fe_terms.append("C(uuid)")
fe_formula = "np.log(speed) ~ " + " + ".join(fe_terms)
fe_model = smf.ols(formula=fe_formula, data=df_dys)
fe_result = fe_model.fit()

# Model 2: pooled with controls, clustered by uuid
control_terms = ["reader_view"]
for col in ["page_id", "num_words", "Flesch_Kincaid", "age", "retake_trial"]:
    if col in df_dys.columns and df_dys[col].dropna().nunique() > 1:
        control_terms.append(f"C({col})" if col == "page_id" else col)
cluster_formula = "np.log(speed) ~ " + " + ".join(control_terms)
cluster_model = smf.ols(formula=cluster_formula, data=df_dys)
cluster_groups = None
if "uuid" in df_dys.columns:
    try:
        cluster_groups = df_dys.loc[cluster_model.data.row_labels, "uuid"]
    except Exception:
        cluster_groups = None
if cluster_groups is not None:
    cluster_codes = pd.factorize(cluster_groups)[0]
    cluster_result = cluster_model.fit(cov_type="cluster", cov_kwds={"groups": cluster_codes})
else:
    cluster_result = cluster_model.fit()

coef = fe_result.params.get("reader_view", np.nan)
se = fe_result.bse.get("reader_view", np.nan)
p_value = fe_result.pvalues.get("reader_view", np.nan)

sec_coef = cluster_result.params.get("reader_view", np.nan)
sec_se = cluster_result.bse.get("reader_view", np.nan)
sec_p_value = cluster_result.pvalues.get("reader_view", np.nan)

# Convert log coefficient to percent change
pct_change = (np.exp(coef) - 1) * 100 if pd.notnull(coef) else np.nan

# Save a compact summary for inspection
summary_lines = []
summary_lines.append("Dyslexia-only descriptive stats (speed):")
summary_lines.append(means.to_string(index=False))
summary_lines.append("")
if paired_result is not None:
    summary_lines.append("Paired summary (within-uuid mean speed, reader_view vs no):")
    for k, v in paired_result.items():
        summary_lines.append(f"{k}: {v}")
    summary_lines.append("")
summary_lines.append("Fixed-effects regression on log(speed):")
summary_lines.append(f"formula: {fe_formula}")
summary_lines.append(f"reader_view coef (log points): {coef}")
summary_lines.append(f"reader_view SE: {se}")
summary_lines.append(f"reader_view p-value: {p_value}")
summary_lines.append(f"Implied % change in speed: {pct_change}")
summary_lines.append("")
summary_lines.append("Pooled regression (clustered by uuid):")
summary_lines.append(f"formula: {cluster_formula}")
summary_lines.append(f"reader_view coef (log points): {sec_coef}")
summary_lines.append(f"reader_view SE: {sec_se}")
summary_lines.append(f"reader_view p-value: {sec_p_value}")
summary_lines.append(f"Implied % change in speed: {(np.exp(sec_coef) - 1) * 100 if pd.notnull(sec_coef) else np.nan}")

with open("analysis_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# Also print key outputs
print("Analysis complete. Key results saved to analysis_output.txt")
