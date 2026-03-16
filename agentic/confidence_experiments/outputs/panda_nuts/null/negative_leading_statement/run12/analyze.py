import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Compute efficiency: nuts opened per second
# Avoid divide-by-zero (seconds should be >0 per metadata, but guard anyway)
df["efficiency"] = df["nuts_opened"] / df["seconds"].replace(0, np.nan)

# Basic checks
n_rows = len(df)

# OLS regression with categorical sex/help
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Also run nonparametric tests for sex/help group differences
# Mann-Whitney U (two-sided) for sex and help
sex_groups = [g["efficiency"].dropna().values for _, g in df.groupby("sex")]
help_groups = [g["efficiency"].dropna().values for _, g in df.groupby("help")]

sex_mwu = None
help_mwu = None
if len(sex_groups) == 2:
    sex_mwu = stats.mannwhitneyu(sex_groups[0], sex_groups[1], alternative="two-sided")
if len(help_groups) == 2:
    help_mwu = stats.mannwhitneyu(help_groups[0], help_groups[1], alternative="two-sided")

# Correlation for age vs efficiency
corr = stats.pearsonr(df["age"], df["efficiency"])

# Prepare results summary
summary = {
    "n_rows": int(n_rows),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "ols_params": model.params.to_dict(),
    "ols_pvalues": model.pvalues.to_dict(),
    "ols_r2": float(model.rsquared),
    "pearson_r": float(corr[0]),
    "pearson_p": float(corr[1]),
    "sex_mwu_p": float(sex_mwu.pvalue) if sex_mwu else None,
    "help_mwu_p": float(help_mwu.pvalue) if help_mwu else None,
}

print(json.dumps(summary, indent=2))
