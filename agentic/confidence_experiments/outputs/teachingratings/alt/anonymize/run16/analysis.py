import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Map columns based on metadata
beauty_col = 'feature6'  # instructor beauty rating
rating_col = 'feature7'  # teaching evaluation score

# Basic stats
n = len(df)

# Correlation
corr = df[[beauty_col, rating_col]].corr().iloc[0, 1]

# Simple OLS
X_simple = sm.add_constant(df[[beauty_col]])
model_simple = sm.OLS(df[rating_col], X_simple).fit(cov_type='HC3')

# Build control variables
# Categorical: feature2 (minority yes/no), feature4 (gender), feature5 (single-credit), feature8 (upper/lower),
# feature9 (native English), feature10 (tenure track)
# Numeric: feature3 (age), feature11 (num students eval), feature12 (enrolled)

cat_cols = ['feature2', 'feature4', 'feature5', 'feature8', 'feature9', 'feature10']
num_cols = ['feature3', 'feature11', 'feature12']

# Create dummies, drop first to avoid multicollinearity
X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
X_num = df[num_cols].copy()

X_control = pd.concat([df[[beauty_col]], X_num, X_cat], axis=1)
X_control = sm.add_constant(X_control)
model_control = sm.OLS(df[rating_col], X_control).fit(cov_type='HC3')

# Extract coefficients and p-values
coef_simple = model_simple.params[beauty_col]
p_simple = model_simple.pvalues[beauty_col]

coef_control = model_control.params[beauty_col]
p_control = model_control.pvalues[beauty_col]

# Effect size: change in rating per 1 SD in beauty
beauty_sd = df[beauty_col].std()
change_1sd = coef_control * beauty_sd

# R-squared
r2_simple = model_simple.rsquared
r2_control = model_control.rsquared

results = {
    'n': n,
    'corr': corr,
    'coef_simple': coef_simple,
    'p_simple': p_simple,
    'coef_control': coef_control,
    'p_control': p_control,
    'change_1sd': change_1sd,
    'r2_simple': r2_simple,
    'r2_control': r2_control,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
