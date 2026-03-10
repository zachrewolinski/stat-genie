import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Rename features to readable names
rename_map = {
    "feature1": "id",
    "feature2": "year",
    "feature3": "name",
    "feature4": "masfem_index",
    "feature5": "min_pressure",
    "feature6": "female_binary",
    "feature7": "category",
    "feature8": "fatalities",
    "feature9": "damage_2013",
    "feature10": "years_elapsed",
    "feature11": "source",
    "feature12": "masfem_mturk",
    "feature13": "max_wind",
    "feature14": "damage_2015",
}

# Only rename columns that exist
cols_present = {k: v for k, v in rename_map.items() if k in df.columns}
df = df.rename(columns=cols_present)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [
    "masfem_index",
    "female_binary",
    "min_pressure",
    "category",
    "fatalities",
    "damage_2013",
    "years_elapsed",
    "masfem_mturk",
    "max_wind",
    "damage_2015",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows missing key variables
needed = ["fatalities", "masfem_index", "female_binary", "category", "max_wind", "min_pressure"]
available_needed = [c for c in needed if c in df.columns]
analysis_df = df.dropna(subset=available_needed).copy()

# Transform outcome
analysis_df["log_fatalities"] = np.log1p(analysis_df["fatalities"])

# Correlations
corr_masfem_fat = analysis_df[["masfem_index", "fatalities"]].corr().iloc[0,1]
corr_masfem_logfat = analysis_df[["masfem_index", "log_fatalities"]].corr().iloc[0,1]

# Group comparison by female_binary
male = analysis_df[analysis_df["female_binary"] == 0]
female = analysis_df[analysis_df["female_binary"] == 1]

def safe_mean(x):
    return float(np.nanmean(x)) if len(x) else np.nan

mean_fatal_male = safe_mean(male["fatalities"])
mean_fatal_female = safe_mean(female["fatalities"])
median_fatal_male = float(np.nanmedian(male["fatalities"])) if len(male) else np.nan
median_fatal_female = float(np.nanmedian(female["fatalities"])) if len(female) else np.nan

# Non-parametric test due to skew
mw_stat, mw_p = stats.mannwhitneyu(male["fatalities"], female["fatalities"], alternative="two-sided")

# OLS regression: log fatalities ~ femininity + controls
# Controls chosen to capture storm severity
X = analysis_df[["masfem_index", "category", "max_wind", "min_pressure"]].copy()
X = sm.add_constant(X)
y = analysis_df["log_fatalities"]

model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")

# Also model using binary female indicator
X2 = analysis_df[["female_binary", "category", "max_wind", "min_pressure"]].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2, missing="drop").fit(cov_type="HC3")

# Pull key stats
coef_masfem = model.params.get("masfem_index")
p_masfem = model.pvalues.get("masfem_index")
coef_female = model2.params.get("female_binary")
p_female = model2.pvalues.get("female_binary")

# Simple bivariate regression
Xb = sm.add_constant(analysis_df[["masfem_index"]])
model_biv = sm.OLS(y, Xb, missing="drop").fit(cov_type="HC3")
coef_biv = model_biv.params.get("masfem_index")
p_biv = model_biv.pvalues.get("masfem_index")

results = {
    "n_rows": int(len(analysis_df)),
    "corr_masfem_fatal": float(corr_masfem_fat),
    "corr_masfem_logfatal": float(corr_masfem_logfat),
    "mean_fatal_male": mean_fatal_male,
    "mean_fatal_female": mean_fatal_female,
    "median_fatal_male": median_fatal_male,
    "median_fatal_female": median_fatal_female,
    "mw_stat": float(mw_stat),
    "mw_p": float(mw_p),
    "coef_masfem_logfat_controls": float(coef_masfem),
    "p_masfem_logfat_controls": float(p_masfem),
    "coef_female_logfat_controls": float(coef_female),
    "p_female_logfat_controls": float(p_female),
    "coef_masfem_logfat_biv": float(coef_biv),
    "p_masfem_logfat_biv": float(p_biv),
}

print(json.dumps(results, indent=2))
