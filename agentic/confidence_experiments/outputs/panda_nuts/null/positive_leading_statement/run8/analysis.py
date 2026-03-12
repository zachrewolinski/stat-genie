import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical types
for col in ["sex", "help", "hammer"]:
    df[col] = df[col].astype("category")

# Efficiency: nuts per second
# Avoid division by zero; seconds min 2.5 per metadata
if (df["seconds"] <= 0).any():
    raise ValueError("Non-positive seconds encountered")

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Center age for interpretability
age_mean = df["age"].mean()
df["age_c"] = df["age"] - age_mean

# Model 1: OLS on efficiency
ols1 = smf.ols("efficiency ~ age_c + sex + help", data=df).fit(cov_type="HC3")

# Model 2: OLS with hammer control
ols2 = smf.ols("efficiency ~ age_c + sex + help + hammer", data=df).fit(cov_type="HC3")

# Poisson regression on nuts_opened with log(seconds) offset (rate model)
# Add small constant if zeros? Poisson can handle zeros.
# Use robust SE (HC3) for potential overdispersion
poisson1 = smf.glm(
    "nuts_opened ~ age_c + sex + help",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit(cov_type="HC3")

poisson2 = smf.glm(
    "nuts_opened ~ age_c + sex + help + hammer",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit(cov_type="HC3")

# Additional tests: correlations and group comparisons
pearson_corr = stats.pearsonr(df["age"], df["efficiency"])
spearman_corr = stats.spearmanr(df["age"], df["efficiency"])

def group_stats(col, label_a, label_b):
    group_a = df.loc[df[col] == label_a, "efficiency"]
    group_b = df.loc[df[col] == label_b, "efficiency"]
    ttest = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
    mannwhitney = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return {
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "n_a": int(group_a.shape[0]),
        "n_b": int(group_b.shape[0]),
        "ttest_stat": float(ttest.statistic),
        "ttest_p": float(ttest.pvalue),
        "mw_stat": float(mannwhitney.statistic),
        "mw_p": float(mannwhitney.pvalue),
    }

sex_stats = group_stats("sex", "f", "m")
help_stats = group_stats("help", "N", "y")

# Collect key results

def summarize(model, label):
    params = model.params
    pvals = model.pvalues
    conf = model.conf_int()
    out = {
        "label": label,
        "n": int(model.nobs),
        "params": params.to_dict(),
        "pvalues": pvals.to_dict(),
        "conf_int": conf.rename(columns={0: "low", 1: "high"}).to_dict("index"),
    }
    return out

results = {
    "age_mean": float(age_mean),
    "efficiency_summary": {
        "mean": float(df["efficiency"].mean()),
        "std": float(df["efficiency"].std()),
        "min": float(df["efficiency"].min()),
        "max": float(df["efficiency"].max()),
    },
    "ols1": summarize(ols1, "efficiency ~ age + sex + help"),
    "ols2": summarize(ols2, "efficiency ~ age + sex + help + hammer"),
    "poisson1": summarize(poisson1, "nuts_opened ~ age + sex + help + offset(log(seconds))"),
    "poisson2": summarize(poisson2, "nuts_opened ~ age + sex + help + hammer + offset(log(seconds))"),
    "age_efficiency_pearson": {"r": float(pearson_corr.statistic), "p": float(pearson_corr.pvalue)},
    "age_efficiency_spearman": {"r": float(spearman_corr.correlation), "p": float(spearman_corr.pvalue)},
    "sex_efficiency_tests": sex_stats,
    "help_efficiency_tests": help_stats,
}

# Save results to a JSON for inspection
import json
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print a brief summary for the console
print("OLS1 summary (robust HC3):")
print(ols1.summary())
print("\nOLS2 summary (robust HC3):")
print(ols2.summary())
print("\nPoisson1 summary (robust HC3):")
print(poisson1.summary())
print("\nPoisson2 summary (robust HC3):")
print(poisson2.summary())
