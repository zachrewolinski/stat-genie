import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Identify columns
# Based on info.json: feature2 = affairs frequency, feature6 = children yes/no

df = _df.copy()

# Basic cleaning: ensure expected columns
required = ['feature2', 'feature6']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Encode children indicator
# feature6 is category 'yes'/'no'
# Create binary: children=1 if yes

df['children'] = df['feature6'].map({'yes': 1, 'no': 0})

# Outcome
outcome = df['feature2'].astype(float)

# Group means
mean_with = outcome[df['children'] == 1].mean()
mean_without = outcome[df['children'] == 0].mean()

# Difference (with - without). Negative means fewer affairs with children.
mean_diff = mean_with - mean_without

# t-test (Welch) using statsmodels
from statsmodels.stats.weightstats import ttest_ind

with_vals = outcome[df['children'] == 1]
without_vals = outcome[df['children'] == 0]

t_stat, p_val, dfree = ttest_ind(with_vals, without_vals, usevar='unequal')

# Regression (OLS) controlling for basic covariates to check robustness
# Use other features where available (numeric) and gender
# feature3 gender, feature4 age, feature5 years married, feature7 religiousness,
# feature8 education, feature9 occupation, feature10 marriage rating

# Prepare dataframe with controls
model_df = df.copy()

# Encode gender
model_df['male'] = model_df['feature3'].map({'male': 1, 'female': 0})

# Ensure numeric columns
num_cols = ['feature2', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']
for c in num_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors='coerce')

# Drop rows with missing data
model_df = model_df.dropna(subset=['feature2', 'children', 'male'] + num_cols[1:])

# OLS
formula = 'feature2 ~ children + male + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
ols = smf.ols(formula=formula, data=model_df).fit()

coef_children = ols.params['children']
se_children = ols.bse['children']

# Convert effect to standardized effect size (Cohen's d) using pooled SD for group diff
# For simple mean diff
n1 = with_vals.shape[0]
n0 = without_vals.shape[0]
var1 = with_vals.var(ddof=1)
var0 = without_vals.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1 + n0 - 2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# Save summary metrics to a json-like text for inspection
summary = {
    'n_total': int(df.shape[0]),
    'n_children_yes': int(n1),
    'n_children_no': int(n0),
    'mean_affairs_children_yes': float(mean_with),
    'mean_affairs_children_no': float(mean_without),
    'mean_diff_yes_minus_no': float(mean_diff),
    't_stat': float(t_stat),
    'p_val': float(p_val),
    'cohen_d': float(cohen_d),
    'ols_children_coef': float(coef_children),
    'ols_children_se': float(se_children),
    'ols_children_p': float(ols.pvalues['children'])
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

