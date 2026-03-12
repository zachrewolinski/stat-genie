import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("panda_nuts.csv")

# Compute efficiency: nuts opened per second
# Avoid division by zero (duration min is 2.5 in metadata)

df = df.copy()

df["efficiency"] = df["feature5"] / df["feature6"]

# Clean categories
# Sex: feature3, Help: feature7
df["sex"] = df["feature3"].astype(str).str.strip()
df["help"] = df["feature7"].astype(str).str.strip()

# Basic info
n_total = len(df)

# Drop rows with missing in key variables
analysis_df = df[["efficiency", "feature2", "sex", "help"]].dropna()

n_used = len(analysis_df)

# Descriptives
sex_counts = analysis_df["sex"].value_counts().to_dict()
help_counts = analysis_df["help"].value_counts().to_dict()

# OLS with robust SE
model = smf.ols("efficiency ~ feature2 + C(sex) + C(help)", data=analysis_df).fit(cov_type="HC3")

# Log1p efficiency as robustness
analysis_df["log_eff"] = np.log1p(analysis_df["efficiency"])
model_log = smf.ols("log_eff ~ feature2 + C(sex) + C(help)", data=analysis_df).fit(cov_type="HC3")

# Age correlation
pearson_r, pearson_p = stats.pearsonr(analysis_df["feature2"], analysis_df["efficiency"])

# Sex group comparison
sex_levels = analysis_df["sex"].unique()
sex_groups = {s: analysis_df.loc[analysis_df["sex"] == s, "efficiency"] for s in sex_levels}

# Help group comparison
help_levels = analysis_df["help"].unique()
help_groups = {h: analysis_df.loc[analysis_df["help"] == h, "efficiency"] for h in help_levels}


def group_test(groups):
    levels = list(groups.keys())
    if len(levels) != 2:
        return None
    a = groups[levels[0]]
    b = groups[levels[1]]
    # t-test with unequal variances
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    # Mann-Whitney U
    try:
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        u_stat, u_p = np.nan, np.nan
    # Cohen's d
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2)) if (na + nb - 2) > 0 else np.nan
    d = (np.mean(a) - np.mean(b)) / pooled if pooled not in (0, np.nan) else np.nan
    return {
        "levels": levels,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "t_p": float(t_p),
        "u_p": float(u_p),
        "cohen_d": float(d),
    }

sex_test = group_test(sex_groups)
help_test = group_test(help_groups)

# Summaries
summary = {
    "n_total": int(n_total),
    "n_used": int(n_used),
    "sex_counts": sex_counts,
    "help_counts": help_counts,
    "efficiency_desc": {
        "mean": float(analysis_df["efficiency"].mean()),
        "std": float(analysis_df["efficiency"].std()),
        "min": float(analysis_df["efficiency"].min()),
        "max": float(analysis_df["efficiency"].max()),
    },
    "model_params": model.params.to_dict(),
    "model_pvalues": model.pvalues.to_dict(),
    "model_r2": float(model.rsquared),
    "model_log_params": model_log.params.to_dict(),
    "model_log_pvalues": model_log.pvalues.to_dict(),
    "model_log_r2": float(model_log.rsquared),
    "age_corr": {"r": float(pearson_r), "p": float(pearson_p)},
    "sex_test": sex_test,
    "help_test": help_test,
}

print(json.dumps(summary, indent=2))
