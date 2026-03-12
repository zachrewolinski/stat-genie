import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


df = pd.read_csv('panda_nuts.csv')

# Standardize column names? ensure expected columns exist
print('Columns:', df.columns.tolist())

# Basic cleaning
# Ensure numeric
for col in ['age','nuts_opened','seconds','female']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Efficiency: nuts per second
if 'nuts_opened' in df.columns and 'seconds' in df.columns:
    df['efficiency'] = df['nuts_opened'] / df['seconds']

# Use sex and help as categorical
if 'sex' in df.columns:
    df['sex'] = df['sex'].astype('category')
if 'help' in df.columns:
    df['help'] = df['help'].astype('category')

# Drop rows with missing key vars
vars_needed = ['efficiency','age','sex','help']
use_cols = [c for c in vars_needed if c in df.columns]
model_df = df.dropna(subset=use_cols).copy()
print('Rows used:', len(model_df))

# OLS with categorical predictors; use cluster-robust SE by chimpanzee if available
formula = 'efficiency ~ age + C(sex) + C(help)'

if 'chimpanzee' in model_df.columns:
    model = smf.ols(formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['chimpanzee']})
else:
    model = smf.ols(formula, data=model_df).fit()

print(model.summary())

# Also test overall significance with ANOVA (no clustering), to gauge effect presence
model_std = smf.ols(formula, data=model_df).fit()
print('ANOVA:')
print(anova_lm(model_std, typ=2))

# Provide effect sizes: partial eta squared from ANOVA
anova = anova_lm(model_std, typ=2)
ss_res = anova.loc['Residual', 'sum_sq']
partial_eta = (anova['sum_sq'] / (anova['sum_sq'] + ss_res)).drop('Residual')
print('Partial eta squared:')
print(partial_eta)

# Save key results for convenience
results = {
    'n': len(model_df),
    'params': model.params.to_dict(),
    'pvalues_cluster': model.pvalues.to_dict(),
    'r2': model.rsquared,
    'anova_p': anova['PR(>F)'].to_dict(),
    'partial_eta_sq': partial_eta.to_dict(),
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print('Saved analysis_results.json')
