import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = "amtl.csv"

df = pd.read_csv(DF_PATH)

# Keep relevant columns
cols = ["num_amtl", "genus", "age", "prob_male", "tooth_class"]
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Missing columns: {missing_cols}")

df_sub = df[cols].dropna().copy()

# Ensure categories
# Statsmodels will use alphabetical ordering for baseline. We will check it.
model = smf.ols(
    "num_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
    data=df_sub,
).fit(cov_type="HC3")

params = model.params
pvalues = model.pvalues

# Identify genus levels used in coefficients
coef_names = list(params.index)

# Collect genus coefficients (non-baseline)
# They will look like C(genus)[T.X]
nonhuman = ["Pan", "Pongo", "Papio"]
coef_map = {g: None for g in nonhuman}
for name in coef_names:
    if name.startswith("C(genus)[T."):
        level = name.split("C(genus)[T.", 1)[1].rstrip("]")
        if level in coef_map:
            coef_map[level] = name

# Determine baseline genus
all_genus_levels = sorted(df_sub["genus"].unique())
# Baseline is the first alphabetically in statsmodels by default
baseline = all_genus_levels[0] if all_genus_levels else None

# Compute average non-human coefficient and test
# Only include those present
present_nonhuman = [g for g in nonhuman if coef_map.get(g)]

avg_coef = None
avg_pvalue = None
if present_nonhuman:
    k = len(present_nonhuman)
    c = np.zeros(len(params))
    for g in present_nonhuman:
        name = coef_map[g]
        idx = coef_names.index(name)
        c[idx] = 1.0 / k
    test = model.t_test(c)
    avg_coef = float(test.effect)
    avg_pvalue = float(test.pvalue)

# Gather pairwise genus coefficients vs baseline
pairwise = {}
for g in present_nonhuman:
    name = coef_map[g]
    pairwise[g] = {
        "coef": float(params[name]),
        "pvalue": float(pvalues[name]),
    }

# Simple descriptive: adjusted mean predictions by genus (marginal over observed covariates)
# For each genus, set genus to that level for all rows, predict, take mean
adj_means = {}
for g in sorted(df_sub["genus"].unique()):
    tmp = df_sub.copy()
    tmp["genus"] = g
    adj_means[g] = float(model.predict(tmp).mean())

# Save analysis summary to JSON for later use
summary = {
    "n_rows": int(df_sub.shape[0]),
    "baseline_genus": baseline,
    "genus_levels": sorted(df_sub["genus"].unique()),
    "pairwise_vs_baseline": pairwise,
    "avg_nonhuman_coef": avg_coef,
    "avg_nonhuman_pvalue": avg_pvalue,
    "adjusted_means": adj_means,
    "model_r2": float(model.rsquared),
    "model_r2_adj": float(model.rsquared_adj),
}

with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
