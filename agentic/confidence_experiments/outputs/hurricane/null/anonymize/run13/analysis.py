import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Map columns for readability based on info.json
COL_ID = "feature1"
COL_YEAR = "feature2"
COL_NAME = "feature3"
COL_MASFEM = "feature4"  # masculinity-femininity index (higher = more feminine)
COL_PRESSURE = "feature5"
COL_FEMALE = "feature6"  # binary gender indicator
COL_CATEGORY = "feature7"
COL_DEATHS = "feature8"
COL_DAMAGE2013 = "feature9"
COL_YEARS_SINCE = "feature10"
COL_SOURCE = "feature11"
COL_MASFEM_MTURK = "feature12"
COL_WIND = "feature13"
COL_DAMAGE2015 = "feature14"

# Basic cleaning: ensure numeric columns are numeric
num_cols = [
    COL_MASFEM, COL_PRESSURE, COL_FEMALE, COL_CATEGORY,
    COL_DEATHS, COL_DAMAGE2013, COL_YEARS_SINCE, COL_MASFEM_MTURK,
    COL_WIND, COL_DAMAGE2015, COL_YEAR
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with missing key fields
key_cols = [COL_MASFEM, COL_DEATHS, COL_PRESSURE, COL_CATEGORY, COL_WIND, COL_YEAR]
reg_df = df.dropna(subset=key_cols).copy()

# Outcome: log1p deaths to reduce skew and handle zeros
reg_df["log_deaths"] = np.log1p(reg_df[COL_DEATHS])

# Model 1: log_deaths ~ masfem + storm intensity controls + year
X1 = reg_df[[COL_MASFEM, COL_CATEGORY, COL_PRESSURE, COL_WIND, COL_YEAR]]
X1 = sm.add_constant(X1)
model1 = sm.OLS(reg_df["log_deaths"], X1).fit(cov_type="HC3")

# Model 2: log_deaths ~ female indicator + controls + year
X2 = reg_df[[COL_FEMALE, COL_CATEGORY, COL_PRESSURE, COL_WIND, COL_YEAR]]
X2 = sm.add_constant(X2)
model2 = sm.OLS(reg_df["log_deaths"], X2).fit(cov_type="HC3")

# Correlations (Spearman) between femininity and deaths
spearman_masfem = reg_df[[COL_MASFEM, COL_DEATHS]].corr(method="spearman").iloc[0,1]
spearman_female = reg_df[[COL_FEMALE, COL_DEATHS]].corr(method="spearman").iloc[0,1]

results = {
    "n": int(reg_df.shape[0]),
    "spearman_masfem_deaths": float(spearman_masfem),
    "spearman_female_deaths": float(spearman_female),
    "model1_masfem_coef": float(model1.params[COL_MASFEM]),
    "model1_masfem_p": float(model1.pvalues[COL_MASFEM]),
    "model1_r2": float(model1.rsquared),
    "model2_female_coef": float(model2.params[COL_FEMALE]),
    "model2_female_p": float(model2.pvalues[COL_FEMALE]),
    "model2_r2": float(model2.rsquared),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
