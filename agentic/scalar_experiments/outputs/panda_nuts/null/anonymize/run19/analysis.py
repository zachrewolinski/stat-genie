import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/panda_nuts/null/anonymize/run19/panda_nuts.csv"

df = pd.read_csv(csv_path)

# Rename for clarity
col_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "helped",
}

df = df.rename(columns=col_map)

# Compute efficiency: nuts per minute
# Avoid division by zero; duration has min 2.5 so safe

df["efficiency_npm"] = df["nuts_opened"] / (df["duration_sec"] / 60.0)

# Basic summaries
summary = {
    "n": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency_npm"].mean()),
    "efficiency_sd": float(df["efficiency_npm"].std(ddof=1)),
}

# Group differences
sex_groups = df.groupby("sex")["efficiency_npm"]
help_groups = df.groupby("helped")["efficiency_npm"]

# t-tests (Welch) for sex and help
sex_vals = [g.values for _, g in sex_groups]
help_vals = [g.values for _, g in help_groups]

# Ensure order for reporting
sex_levels = list(sex_groups.groups.keys())
help_levels = list(help_groups.groups.keys())

sex_t, sex_p = stats.ttest_ind(sex_vals[0], sex_vals[1], equal_var=False)
help_t, help_p = stats.ttest_ind(help_vals[0], help_vals[1], equal_var=False)

# Correlation with age
age_corr_r, age_corr_p = stats.pearsonr(df["age"], df["efficiency_npm"])

# Multivariate model: efficiency ~ age + sex + helped
# Use robust standard errors (HC3) due to potential heteroskedasticity
model = smf.ols("efficiency_npm ~ age + C(sex) + C(helped)", data=df).fit(cov_type="HC3")

model_summary = {
    "params": model.params.to_dict(),
    "pvalues": model.pvalues.to_dict(),
    "conf_int": {k: [float(v[0]), float(v[1])] for k, v in model.conf_int().iterrows()},
    "r2": float(model.rsquared),
    "adj_r2": float(model.rsquared_adj),
}

# Effect sizes for group differences (Hedges g)

def hedges_g(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / dof
    if pooled <= 0:
        return np.nan
    g = (np.mean(x) - np.mean(y)) / np.sqrt(pooled)
    # small sample correction
    correction = 1 - (3 / (4 * dof - 1))
    return float(g * correction)

sex_g = hedges_g(sex_vals[0], sex_vals[1])
help_g = hedges_g(help_vals[0], help_vals[1])

results = {
    "summary": summary,
    "sex_levels": sex_levels,
    "help_levels": help_levels,
    "sex_t": float(sex_t),
    "sex_p": float(sex_p),
    "help_t": float(help_t),
    "help_p": float(help_p),
    "sex_g": sex_g,
    "help_g": help_g,
    "age_corr_r": float(age_corr_r),
    "age_corr_p": float(age_corr_p),
    "model": model_summary,
}

with open("/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/panda_nuts/null/anonymize/run19/analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
