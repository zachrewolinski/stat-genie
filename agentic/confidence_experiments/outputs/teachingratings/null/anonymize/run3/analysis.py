import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Identify columns from metadata
beauty = "feature6"
rating = "feature7"

# Basic stats
n = len(df)

# Correlation
corr = df[[beauty, rating]].corr().iloc[0,1]

# Simple OLS: rating ~ beauty
X_simple = sm.add_constant(df[[beauty]])
model_simple = sm.OLS(df[rating], X_simple).fit()

# Build multivariate model with available controls
# Convert categorical to dummies
categorical_cols = ["feature2", "feature4", "feature5", "feature8", "feature9", "feature10"]
num_cols = ["feature3", "feature11", "feature12"]

# Some columns might be non-numeric, ensure correct types
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

X = df[[beauty] + num_cols + categorical_cols]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X = sm.add_constant(X)
model_full = sm.OLS(df[rating], X).fit()

# Extract key stats
simple_coef = model_simple.params[beauty]
simple_p = model_simple.pvalues[beauty]
full_coef = model_full.params[beauty]
full_p = model_full.pvalues[beauty]

# Also compute standardized effect: beauty z, rating z
beauty_z = (df[beauty] - df[beauty].mean()) / df[beauty].std(ddof=0)
rating_z = (df[rating] - df[rating].mean()) / df[rating].std(ddof=0)
X_z = sm.add_constant(beauty_z)
model_z = sm.OLS(rating_z, X_z).fit()
std_beta = model_z.params[beauty]

results = {
    "n": n,
    "corr": corr,
    "simple_coef": simple_coef,
    "simple_p": simple_p,
    "simple_r2": model_simple.rsquared,
    "full_coef": full_coef,
    "full_p": full_p,
    "full_r2": model_full.rsquared,
    "std_beta": std_beta,
}

print(json.dumps(results, indent=2))
