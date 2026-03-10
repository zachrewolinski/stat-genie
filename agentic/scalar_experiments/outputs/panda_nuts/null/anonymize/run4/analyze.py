import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("panda_nuts.csv")

# Map columns
id_col = "feature1"
age_col = "feature2"
sex_col = "feature3"
hammer_col = "feature4"
nuts_col = "feature5"
dur_col = "feature6"
help_col = "feature7"

# Efficiency: nuts opened per second
# Avoid division by zero; if any zero duration, set to NaN

df = df.copy()
df["efficiency"] = df[nuts_col] / df[dur_col].replace(0, np.nan)

# Clean categories
for col in [sex_col, hammer_col, help_col]:
    df[col] = df[col].astype("category")

# Basic sanity

# Model: efficiency ~ age + sex + help + hammer (control)
formula = "efficiency ~ {} + C({}) + C({}) + C({})".format(age_col, sex_col, help_col, hammer_col)
model = smf.ols(formula, data=df).fit(cov_type="HC3")

# Also run without hammer as sensitivity
formula2 = "efficiency ~ {} + C({}) + C({})".format(age_col, sex_col, help_col)
model2 = smf.ols(formula2, data=df).fit(cov_type="HC3")

# Collect key results

def extract(model):
    params = model.params
    pvals = model.pvalues
    conf = model.conf_int()
    out = {}
    for key in params.index:
        out[key] = {
            "coef": float(params[key]),
            "p": float(pvals[key]),
            "ci_low": float(conf.loc[key, 0]),
            "ci_high": float(conf.loc[key, 1]),
        }
    return out

results = {
    "n": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "model_with_hammer": extract(model),
    "model_without_hammer": extract(model2),
    "r2_with_hammer": float(model.rsquared),
    "r2_without_hammer": float(model2.rsquared),
}

print(results)
