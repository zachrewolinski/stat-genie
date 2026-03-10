import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: ensure categorical columns are treated as category
categorical_cols = [
    'minority', 'gender', 'credits', 'division', 'native', 'tenure'
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Outcome and predictor
# eval (teaching evaluation) vs beauty

results = {}

# Simple correlation
corr = df['beauty'].corr(df['eval'])
results['corr'] = corr

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()
results['simple'] = {
    'coef': model_simple.params['beauty'],
    'pvalue': model_simple.pvalues['beauty'],
    'ci_low': model_simple.conf_int().loc['beauty'][0],
    'ci_high': model_simple.conf_int().loc['beauty'][1],
    'r2': model_simple.rsquared,
}

# Multiple regression with controls
controls = ['age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'allstudents']
# Some datasets include allstudents and students; potential multicollinearity but keep for transparency.
# Build formula with categorical terms
formula_terms = ['beauty']
for c in controls:
    if c in df.columns:
        if pd.api.types.is_categorical_dtype(df[c]):
            formula_terms.append(f'C({c})')
        else:
            formula_terms.append(c)

formula = 'eval ~ ' + ' + '.join(formula_terms)
model_full = smf.ols(formula, data=df).fit()
results['full'] = {
    'coef': model_full.params['beauty'],
    'pvalue': model_full.pvalues['beauty'],
    'ci_low': model_full.conf_int().loc['beauty'][0],
    'ci_high': model_full.conf_int().loc['beauty'][1],
    'r2': model_full.rsquared,
}

# Save results
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
