import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Rename columns to meaningful names based on value ranges/descriptions
rename = {
    "ndam": "id",
    "wind": "year",
    "alldeaths": "name",
    "category": "femininity_index",
    "ndam15": "min_pressure",
    "masfem_mturk": "female_binary",
    "gender_mf": "ss_category",
    "name": "deaths",
    "elapsedyrs": "damage_2013",
    "masfem": "years_elapsed",
    "min": "source",
    "ind": "mturk_femininity",
    "year": "wind_speed",
    "source": "damage_2015",
}

# Ensure all columns exist
missing = [c for c in rename if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.rename(columns=rename)

# Basic checks
print("Rows:", len(df))

# Prepare variables
# deaths can be zero; use log1p

df["log_deaths"] = np.log1p(df["deaths"].astype(float))

# Drop rows with missing key values
model_df = df.dropna(subset=["log_deaths", "femininity_index", "wind_speed", "min_pressure", "ss_category"])

# Standardize severity variables for interaction (optional)
for col in ["wind_speed", "min_pressure", "ss_category"]:
    model_df[col + "_z"] = (model_df[col] - model_df[col].mean()) / model_df[col].std()

# OLS main effects
formula_main = "log_deaths ~ femininity_index + wind_speed_z + min_pressure_z + ss_category_z"
model_main = smf.ols(formula_main, data=model_df).fit(cov_type="HC3")
print("\nMain effects model:")
print(model_main.summary())

# Interaction with severity (wind speed)
formula_int = "log_deaths ~ femininity_index * wind_speed_z + min_pressure_z + ss_category_z"
model_int = smf.ols(formula_int, data=model_df).fit(cov_type="HC3")
print("\nInteraction model (femininity x wind_speed):")
print(model_int.summary())

# Simple correlation
corr = model_df[["femininity_index", "log_deaths"]].corr().iloc[0,1]
print("\nCorrelation femininity vs log_deaths:", corr)

# Female binary comparison
if "female_binary" in model_df.columns:
    grouped = model_df.groupby("female_binary")["log_deaths"].agg(['mean','count'])
    print("\nLog deaths by female_binary:")
    print(grouped)

# t-test for female_binary
try:
    import scipy.stats as stats
    g0 = model_df.loc[model_df["female_binary"]==0, "log_deaths"]
    g1 = model_df.loc[model_df["female_binary"]==1, "log_deaths"]
    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    print("\nT-test log_deaths female vs male:", tstat, pval)
except Exception as e:
    print("T-test failed", e)

