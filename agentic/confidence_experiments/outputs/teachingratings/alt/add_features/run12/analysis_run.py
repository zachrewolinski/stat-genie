import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic columns inspection
print("columns:", list(df.columns))
print("shape:", df.shape)

# Ensure key columns exist
# Use eval (rating) and beauty

# Drop rows with missing values in relevant columns
# Identify available covariates from canonical dataset
candidate_covariates = [
    "age",
    "gender",
    "division",
    "native",
    "tenure",
    "students",
    "credits",
    "minority",
]

available_covariates = [c for c in candidate_covariates if c in df.columns]

# Build formula: eval ~ beauty + covariates
formula = "eval ~ beauty"
for c in available_covariates:
    formula += f" + C({c})" if df[c].dtype == object else f" + {c}"

# Only keep rows with non-missing for involved columns
cols = ["eval", "beauty"] + available_covariates
model_df = df[cols].dropna()

print("model_df shape:", model_df.shape)

# Fit OLS
model = smf.ols(formula, data=model_df).fit()
print(model.summary())

# Extract beauty coefficient and p-value
beauty_coef = model.params.get("beauty")
beauty_p = model.pvalues.get("beauty")
print("beauty_coef", beauty_coef)
print("beauty_p", beauty_p)

# Also compute simple correlation
corr = model_df[["eval", "beauty"]].corr().iloc[0,1]
print("corr_eval_beauty", corr)

# For effect size: standard deviation change in eval for 1 SD in beauty
beauty_sd = model_df["beauty"].std()
impact_1sd = beauty_coef * beauty_sd if beauty_coef is not None else np.nan
print("impact_1sd_eval", impact_1sd)
