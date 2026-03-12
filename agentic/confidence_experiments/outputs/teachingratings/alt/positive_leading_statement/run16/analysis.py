import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "teachingratings.csv"


df = pd.read_csv(DATA_PATH)

# Ensure categorical types
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for c in cat_cols:
    df[c] = df[c].astype("category")

# Basic correlation
corr, corr_p = stats.pearsonr(df["beauty"], df["eval"])

# Simple OLS
m_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Controlled OLS
formula = "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents"
m_ctrl = smf.ols(formula, data=df).fit(cov_type="HC3")

beauty_sd = df["beauty"].std()

results = {
    "n": int(df.shape[0]),
    "corr": corr,
    "corr_p": corr_p,
    "simple_coef": m_simple.params["beauty"],
    "simple_p": m_simple.pvalues["beauty"],
    "simple_ci": m_simple.conf_int().loc["beauty"].tolist(),
    "ctrl_coef": m_ctrl.params["beauty"],
    "ctrl_p": m_ctrl.pvalues["beauty"],
    "ctrl_ci": m_ctrl.conf_int().loc["beauty"].tolist(),
    "beauty_sd": beauty_sd,
    "ctrl_effect_1sd": m_ctrl.params["beauty"] * beauty_sd,
}

print(json.dumps(results, indent=2))
