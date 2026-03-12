import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("hurricane.csv")

# Column mapping based on info.json
col = {
    "id": "feature1",
    "year": "feature2",
    "name": "feature3",
    "masfem": "feature4",
    "min_pressure": "feature5",
    "female_binary": "feature6",
    "category": "feature7",
    "deaths": "feature8",
    "damage_2013": "feature9",
    "years_elapsed": "feature10",
    "source": "feature11",
    "mturk_masfem": "feature12",
    "max_wind": "feature13",
    "damage_2015": "feature14",
}

# Basic cleaning
_df = _df.copy()

# Ensure numeric
num_cols = [
    col["year"], col["masfem"], col["min_pressure"], col["female_binary"], col["category"],
    col["deaths"], col["damage_2013"], col["years_elapsed"], col["mturk_masfem"],
    col["max_wind"], col["damage_2015"],
]
for c in num_cols:
    _df[c] = pd.to_numeric(_df[c], errors="coerce")

# Outcome transform
_df["log1p_deaths"] = np.log1p(_df[col["deaths"]])

# Group comparison by binary gender
male = _df[_df[col["female_binary"]] == 0][col["deaths"]]
female = _df[_df[col["female_binary"]] == 1][col["deaths"]]

# Two-sided t-test (Welch) on log1p deaths to handle skew
male_log = np.log1p(male.dropna())
female_log = np.log1p(female.dropna())

ttest = stats.ttest_ind(male_log, female_log, equal_var=False, nan_policy="omit")

# Correlation between masfem and log1p deaths
corr = stats.pearsonr(_df[col["masfem"]], _df["log1p_deaths"])

# Regression model: log1p deaths ~ masfem + controls
# Controls chosen to proxy storm severity and exposure
# Use damage_2013 as normalized damage exposure proxy
reg_df = _df[[
    "log1p_deaths",
    col["masfem"],
    col["category"],
    col["min_pressure"],
    col["max_wind"],
    col["damage_2013"],
    col["year"],
]].dropna()

formula = (
    "log1p_deaths ~ masfem + category + min_pressure + max_wind + damage_2013 + year"
)
reg_df = reg_df.rename(columns={
    col["masfem"]: "masfem",
    col["category"]: "category",
    col["min_pressure"]: "min_pressure",
    col["max_wind"]: "max_wind",
    col["damage_2013"]: "damage_2013",
    col["year"]: "year",
})

model = smf.ols(formula, data=reg_df).fit(cov_type="HC3")

# Alternative model with mturk masfem
alt_df = _df[[
    "log1p_deaths",
    col["mturk_masfem"],
    col["category"],
    col["min_pressure"],
    col["max_wind"],
    col["damage_2013"],
    col["year"],
]].dropna()
alt_df = alt_df.rename(columns={
    col["mturk_masfem"]: "mturk_masfem",
    col["category"]: "category",
    col["min_pressure"]: "min_pressure",
    col["max_wind"]: "max_wind",
    col["damage_2013"]: "damage_2013",
    col["year"]: "year",
})
alt_formula = (
    "log1p_deaths ~ mturk_masfem + category + min_pressure + max_wind + damage_2013 + year"
)
alt_model = smf.ols(alt_formula, data=alt_df).fit(cov_type="HC3")

# Collect results
results = {
    "n_total": int(len(_df)),
    "n_reg": int(len(reg_df)),
    "mean_deaths_male": float(male.mean()),
    "mean_deaths_female": float(female.mean()),
    "ttest_log1p": {
        "stat": float(ttest.statistic),
        "pvalue": float(ttest.pvalue),
    },
    "corr_masfem_log1p": {
        "r": float(corr[0]),
        "pvalue": float(corr[1]),
    },
    "reg_masfem": {
        "coef": float(model.params.get("masfem", np.nan)),
        "pvalue": float(model.pvalues.get("masfem", np.nan)),
        "conf_int_low": float(model.conf_int().loc["masfem", 0]),
        "conf_int_high": float(model.conf_int().loc["masfem", 1]),
        "r2": float(model.rsquared),
    },
    "reg_mturk_masfem": {
        "coef": float(alt_model.params.get("mturk_masfem", np.nan)),
        "pvalue": float(alt_model.pvalues.get("mturk_masfem", np.nan)),
        "conf_int_low": float(alt_model.conf_int().loc["mturk_masfem", 0]),
        "conf_int_high": float(alt_model.conf_int().loc["mturk_masfem", 1]),
        "r2": float(alt_model.rsquared),
    },
}

print(results)
