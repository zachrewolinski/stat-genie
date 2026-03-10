import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Identify columns
print('Columns:', df.columns.tolist())

# Basic summaries
summary = df.describe(include='all')
print('\nSummary (selected):')
print(summary[['beauty', 'allstudents']])

# Correlation
beauty = df['beauty']
ratings = df['allstudents']
pearson_r, pearson_p = stats.pearsonr(beauty, ratings)
spearman_r, spearman_p = stats.spearmanr(beauty, ratings)
print(f"\nPearson r: {pearson_r:.4f}, p={pearson_p:.4g}")
print(f"Spearman r: {spearman_r:.4f}, p={spearman_p:.4g}")

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()
print('\nSimple OLS: allstudents ~ beauty')
print(model_simple.summary())

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print('\nCategorical columns:', cat_cols)

# Choose control variables: exclude obvious identifiers / high-cardinality numeric columns
# Heuristic: drop numeric columns with near-unique values (likely IDs)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols_no_y = [c for c in num_cols if c not in ['allstudents', 'beauty']]
unique_counts = {c: df[c].nunique() for c in num_cols_no_y}

likely_id = [c for c, n in unique_counts.items() if n > 0.5 * len(df)]
print('\nNumeric columns (excluding y and beauty):', num_cols_no_y)
print('Unique counts:', unique_counts)
print('Likely ID-like numeric columns:', likely_id)

controls = [c for c in num_cols_no_y if c not in likely_id]

# Add categorical controls (excluding any potential high-cardinality if present)
cat_controls = [c for c in cat_cols]

# Build formula
control_terms = []
control_terms += controls
control_terms += [f'C({c})' for c in cat_controls]

formula = 'allstudents ~ beauty'
if control_terms:
    formula += ' + ' + ' + '.join(control_terms)

print('\nAdjusted formula:', formula)

model_adj = smf.ols(formula, data=df).fit()
print('\nAdjusted OLS summary (beauty effect):')
print(model_adj.summary())

# Extract beauty coefficient and CI
coef = model_adj.params['beauty']
conf_int = model_adj.conf_int().loc['beauty'].tolist()
print(f"\nAdjusted beauty coef: {coef:.4f}")
print(f"Adjusted beauty 95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
print(f"Adjusted beauty p-value: {model_adj.pvalues['beauty']:.4g}")

# Simple model CI
coef_s = model_simple.params['beauty']
conf_int_s = model_simple.conf_int().loc['beauty'].tolist()
print(f"\nSimple beauty coef: {coef_s:.4f}")
print(f"Simple beauty 95% CI: [{conf_int_s[0]:.4f}, {conf_int_s[1]:.4f}]")
print(f"Simple beauty p-value: {model_simple.pvalues['beauty']:.4g}")
