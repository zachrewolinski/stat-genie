import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = "hurricane.csv"

df = pd.read_csv(path)

# Focus on relevant columns
cols = ["masfem", "gender_mf", "alldeaths", "wind", "min", "category", "ndam", "ndam15", "year"]

df = df[cols]

# Drop rows with missing values in key variables
key_vars = ["masfem", "alldeaths", "wind", "min", "category"]

df_clean = df.dropna(subset=key_vars).copy()

# Create log deaths
# Add 1 to handle zero deaths

df_clean["log_deaths"] = np.log1p(df_clean["alldeaths"])

# Basic stats
n = len(df_clean)

# Correlation between masfem and log deaths
corr, corr_p = stats.pearsonr(df_clean["masfem"], df_clean["log_deaths"])

# OLS regression: log_deaths ~ masfem + wind + min + category + year
X = df_clean[["masfem", "wind", "min", "category", "year"]]
X = sm.add_constant(X)
model = sm.OLS(df_clean["log_deaths"], X).fit()

# Alternative: using binary gender_mf (if available)
alt = None
if df_clean["gender_mf"].notna().all():
    X2 = df_clean[["gender_mf", "wind", "min", "category", "year"]]
    X2 = sm.add_constant(X2)
    alt = sm.OLS(df_clean["log_deaths"], X2).fit()

# Compile key results
results = {
    "n": n,
    "corr_masfem_logdeaths": corr,
    "corr_p": corr_p,
    "ols_masfem_coef": model.params["masfem"],
    "ols_masfem_p": model.pvalues["masfem"],
    "ols_masfem_ci": model.conf_int().loc["masfem"].tolist(),
    "ols_r2": model.rsquared,
}

if alt is not None:
    results.update({
        "ols_gender_mf_coef": alt.params["gender_mf"],
        "ols_gender_mf_p": alt.pvalues["gender_mf"],
        "ols_gender_mf_ci": alt.conf_int().loc["gender_mf"].tolist(),
        "ols_gender_mf_r2": alt.rsquared,
    })

print(json.dumps(results, indent=2))

# Save detailed summary for review
with open("analysis_summary.txt", "w") as f:
    f.write(model.summary().as_text())
    if alt is not None:
        f.write("\n\n---\n\n")
        f.write(alt.summary().as_text())
