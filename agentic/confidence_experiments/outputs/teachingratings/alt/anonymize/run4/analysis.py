import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Variables based on info.json
beauty = "feature6"  # instructor beauty (mean-centered)
rating = "feature7"  # teaching evaluation score

# Basic cleaning: drop rows with missing values in relevant columns
base_cols = [beauty, rating]

# Define control variables from metadata
control_cols = [
    "feature2",  # minority (yes/no)
    "feature3",  # age
    "feature4",  # gender
    "feature5",  # single-credit elective
    "feature8",  # upper/lower division
    "feature9",  # native English speaker
    "feature10", # tenure track
    "feature11", # number of students in evaluation
    "feature12", # number enrolled
]

used_cols = base_cols + control_cols

df_used = df[used_cols].dropna().copy()

# Encode categorical variables
cat_cols = [
    "feature2",
    "feature4",
    "feature5",
    "feature8",
    "feature9",
    "feature10",
]

df_encoded = pd.get_dummies(df_used, columns=cat_cols, drop_first=True)

# Build design matrix for multivariate regression
X = df_encoded.drop(columns=[rating])
X = sm.add_constant(X)

y = df_encoded[rating]

# OLS with robust (HC3) standard errors
model = sm.OLS(y, X).fit(cov_type="HC3")

# Simple regression (beauty only)
X_simple = sm.add_constant(df_used[[beauty]])
model_simple = sm.OLS(df_used[rating], X_simple).fit(cov_type="HC3")

# Correlation
corr = df_used[[beauty, rating]].corr().iloc[0, 1]

# Standardized effect from multivariate model
# Standardize beauty and rating to interpret beta
beauty_std = (df_used[beauty] - df_used[beauty].mean()) / df_used[beauty].std(ddof=0)
rating_std = (df_used[rating] - df_used[rating].mean()) / df_used[rating].std(ddof=0)

X_std = pd.concat([
    beauty_std.rename(beauty),
    df_used.drop(columns=[beauty, rating]),
], axis=1)
X_std = pd.get_dummies(X_std, columns=cat_cols, drop_first=True)
X_std = sm.add_constant(X_std)

model_std = sm.OLS(rating_std, X_std).fit(cov_type="HC3")

results = {
    "n": int(df_used.shape[0]),
    "corr": float(corr),
    "simple_coef": float(model_simple.params[beauty]),
    "simple_p": float(model_simple.pvalues[beauty]),
    "simple_ci": [float(c) for c in model_simple.conf_int().loc[beauty].tolist()],
    "multi_coef": float(model.params[beauty]),
    "multi_p": float(model.pvalues[beauty]),
    "multi_ci": [float(c) for c in model.conf_int().loc[beauty].tolist()],
    "multi_std_coef": float(model_std.params[beauty]),
    "multi_std_p": float(model_std.pvalues[beauty]),
    "multi_std_ci": [float(c) for c in model_std.conf_int().loc[beauty].tolist()],
    "r2": float(model.rsquared),
    "r2_simple": float(model_simple.rsquared),
}

print(json.dumps(results, indent=2))
