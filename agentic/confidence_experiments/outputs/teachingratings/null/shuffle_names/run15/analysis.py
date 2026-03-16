import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Basic columns
outcome = "allstudents"
beauty = "beauty"

# Helper: identify numeric controls (drop outcome/beauty and high-cardinality IDs)
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Drop outcome and beauty from controls
num_controls = [c for c in num_cols if c not in {outcome, beauty}]

# Drop likely ID-like columns (very high cardinality)
high_cardinality = []
for c in num_controls:
    if df[c].nunique(dropna=True) > 0.9 * len(df):
        high_cardinality.append(c)
num_controls = [c for c in num_controls if c not in high_cardinality]

# Candidate categorical controls
cat_controls = [
    c for c in df.columns
    if df[c].dtype == object and c not in {outcome, beauty}
]

# Bivariate stats
r, r_p = stats.pearsonr(df[beauty], df[outcome])

model_simple = smf.ols(f"{outcome} ~ {beauty}", data=df).fit(cov_type="HC3")

# Multivariate model with controls
control_terms = []
control_terms += num_controls
control_terms += [f"C({c})" for c in cat_controls]

if control_terms:
    formula = f"{outcome} ~ {beauty} + " + " + ".join(control_terms)
else:
    formula = f"{outcome} ~ {beauty}"

model_controls = smf.ols(formula, data=df).fit(cov_type="HC3")

# Effect sizes
beauty_sd = df[beauty].std()
outcome_sd = df[outcome].std()
coef_simple = model_simple.params[beauty]
coef_controls = model_controls.params[beauty]

std_effect_simple = coef_simple * beauty_sd / outcome_sd
std_effect_controls = coef_controls * beauty_sd / outcome_sd

# 95% CI
ci_simple = model_simple.conf_int().loc[beauty].tolist()
ci_controls = model_controls.conf_int().loc[beauty].tolist()

summary = {
    "n": len(df),
    "beauty_sd": beauty_sd,
    "outcome_sd": outcome_sd,
    "pearson_r": r,
    "pearson_p": r_p,
    "simple": {
        "coef": coef_simple,
        "p": model_simple.pvalues[beauty],
        "ci_low": ci_simple[0],
        "ci_high": ci_simple[1],
        "r2": model_simple.rsquared,
        "std_effect": std_effect_simple,
    },
    "controls": {
        "coef": coef_controls,
        "p": model_controls.pvalues[beauty],
        "ci_low": ci_controls[0],
        "ci_high": ci_controls[1],
        "r2": model_controls.rsquared,
        "std_effect": std_effect_controls,
        "num_controls": num_controls,
        "cat_controls": cat_controls,
        "formula": formula,
    },
}

print(json.dumps(summary, indent=2))
