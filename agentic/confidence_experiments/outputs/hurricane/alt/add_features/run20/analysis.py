import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Focus on relevant columns
cols = ["alldeaths", "masfem", "gender_mf", "wind", "min", "category", "year"]
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df[cols].dropna().copy()

# Transformations
df["log_deaths"] = np.log1p(df["alldeaths"])

results = {}
results["n"] = int(df.shape[0])
results["alldeaths_summary"] = {
    "mean": float(df["alldeaths"].mean()),
    "median": float(df["alldeaths"].median()),
    "max": float(df["alldeaths"].max()),
}
results["masfem_summary"] = {
    "mean": float(df["masfem"].mean()),
    "std": float(df["masfem"].std()),
    "min": float(df["masfem"].min()),
    "max": float(df["masfem"].max()),
}

# OLS with robust SE
m1 = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=df).fit(cov_type="HC3")
results["ols_log_m1"] = {
    "coef_masfem": float(m1.params.get("masfem", np.nan)),
    "p_masfem": float(m1.pvalues.get("masfem", np.nan)),
    "r2": float(m1.rsquared),
}

# Interaction with wind
m2 = smf.ols("log_deaths ~ masfem*wind + min + category + year", data=df).fit(cov_type="HC3")
results["ols_log_m2"] = {
    "coef_masfem": float(m2.params.get("masfem", np.nan)),
    "p_masfem": float(m2.pvalues.get("masfem", np.nan)),
    "coef_interaction": float(m2.params.get("masfem:wind", np.nan)),
    "p_interaction": float(m2.pvalues.get("masfem:wind", np.nan)),
    "r2": float(m2.rsquared),
}

# Binary gender
m3 = smf.ols("log_deaths ~ gender_mf + wind + min + category + year", data=df).fit(cov_type="HC3")
results["ols_log_m3_gender"] = {
    "coef_gender_female": float(m3.params.get("gender_mf", np.nan)),
    "p_gender_female": float(m3.pvalues.get("gender_mf", np.nan)),
    "r2": float(m3.rsquared),
}

# Poisson GLM for counts
m4 = smf.glm(
    "alldeaths ~ masfem + wind + min + category + year",
    data=df,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")
results["poisson_m4"] = {
    "coef_masfem": float(m4.params.get("masfem", np.nan)),
    "p_masfem": float(m4.pvalues.get("masfem", np.nan)),
}

# Negative binomial GLM for overdispersed counts
m5 = smf.glm(
    "alldeaths ~ masfem + wind + min + category + year",
    data=df,
    family=sm.families.NegativeBinomial(),
).fit(cov_type="HC3")
results["negbin_m5"] = {
    "coef_masfem": float(m5.params.get("masfem", np.nan)),
    "p_masfem": float(m5.pvalues.get("masfem", np.nan)),
}

# Simple correlation
results["corr_masfem_alldeaths"] = float(df["masfem"].corr(df["alldeaths"]))

# Sensitivity: drop top 2 deadliest storms
df_sens = df.sort_values("alldeaths", ascending=False).iloc[2:].copy()
df_sens["log_deaths"] = np.log1p(df_sens["alldeaths"])

m1_s = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=df_sens).fit(cov_type="HC3")
m4_s = smf.glm(
    "alldeaths ~ masfem + wind + min + category + year",
    data=df_sens,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")

results["sensitivity_drop2"] = {
    "n": int(df_sens.shape[0]),
    "ols_log_coef_masfem": float(m1_s.params.get("masfem", np.nan)),
    "ols_log_p_masfem": float(m1_s.pvalues.get("masfem", np.nan)),
    "poisson_coef_masfem": float(m4_s.params.get("masfem", np.nan)),
    "poisson_p_masfem": float(m4_s.pvalues.get("masfem", np.nan)),
}

# Save results for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
