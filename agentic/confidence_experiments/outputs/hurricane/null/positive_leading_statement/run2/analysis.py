import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "hurricane.csv"

df = pd.read_csv(path)

# Prepare variables
# Outcome: log1p deaths

df["log_deaths"] = np.log1p(df["alldeaths"].astype(float))

# Damage adjustment
if "ndam15" in df.columns:
    df["log_ndam15"] = np.log1p(df["ndam15"].astype(float))
else:
    df["log_ndam15"] = np.nan

# Keep relevant columns
cols = [
    "log_deaths",
    "masfem",
    "gender_mf",
    "wind",
    "min",
    "category",
    "log_ndam15",
]

analysis_df = df[cols].dropna()

# Basic correlations
spearman_masfem = stats.spearmanr(analysis_df["masfem"], analysis_df["alldeaths"] if "alldeaths" in analysis_df else df.loc[analysis_df.index, "alldeaths"], nan_policy="omit")
pearson_masfem = stats.pearsonr(analysis_df["masfem"], analysis_df["log_deaths"])

# OLS models with robust SEs
# Model 1: log deaths ~ masfem
model1 = smf.ols("log_deaths ~ masfem", data=analysis_df).fit(cov_type="HC3")

# Model 2: log deaths ~ masfem + intensity controls (wind, min, category) + damage
model2 = smf.ols(
    "log_deaths ~ masfem + wind + min + category + log_ndam15",
    data=analysis_df,
).fit(cov_type="HC3")

# Model 3: log deaths ~ gender_mf + intensity controls + damage
model3 = smf.ols(
    "log_deaths ~ gender_mf + wind + min + category + log_ndam15",
    data=analysis_df,
).fit(cov_type="HC3")

# Summaries
results = {
    "n": int(len(analysis_df)),
    "spearman_masfem_r": float(spearman_masfem.correlation),
    "spearman_masfem_p": float(spearman_masfem.pvalue),
    "pearson_masfem_r": float(pearson_masfem.statistic),
    "pearson_masfem_p": float(pearson_masfem.pvalue),
    "model1": {
        "coef": float(model1.params.get("masfem", np.nan)),
        "p": float(model1.pvalues.get("masfem", np.nan)),
        "r2": float(model1.rsquared),
    },
    "model2": {
        "coef": float(model2.params.get("masfem", np.nan)),
        "p": float(model2.pvalues.get("masfem", np.nan)),
        "r2": float(model2.rsquared),
    },
    "model3": {
        "coef": float(model3.params.get("gender_mf", np.nan)),
        "p": float(model3.pvalues.get("gender_mf", np.nan)),
        "r2": float(model3.rsquared),
    },
}

print(json.dumps(results, indent=2))
