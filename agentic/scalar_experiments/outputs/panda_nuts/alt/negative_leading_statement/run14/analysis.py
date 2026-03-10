import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure categorical columns as category
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Define efficiency as nuts_opened per second
# Avoid divide by zero just in case
if (df['seconds'] <= 0).any():
    df = df[df['seconds'] > 0].copy()

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Descriptives
summary = df[['efficiency','nuts_opened','seconds','age']].describe().T

# OLS model for efficiency
# Include hammer as control because tool type may affect efficiency
# Also include chimpanzee as a random effect? Not in OLS; we can cluster by individual
# Use robust (HC3) and cluster by chimpanzee to account for repeated measures
model_formula = 'efficiency ~ age + C(sex) + C(help) + C(hammer)'
ols = smf.ols(model_formula, data=df).fit(cov_type='HC3')

# Cluster-robust by chimpanzee
cluster = smf.ols(model_formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

# Also model nuts_opened with offset seconds using Poisson/NegBin? We'll try GLM Poisson with offset
# This models rate directly.
# Add a small constant to seconds to avoid log(0)
rate_df = df.copy()
rate_df['log_seconds'] = np.log(rate_df['seconds'])

poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help) + C(hammer)',
                  data=rate_df,
                  family=sm.families.Poisson(),
                  offset=rate_df['log_seconds']).fit(cov_type='HC3')

# Collect key results

def extract_terms(result, terms):
    rows = []
    for term in terms:
        if term in result.params.index:
            rows.append({
                'term': term,
                'coef': result.params[term],
                'se': result.bse[term],
                'p': result.pvalues[term]
            })
    return pd.DataFrame(rows)

# Terms of interest
terms_ols = ['age']
# C(sex)[T.m] if f is baseline, C(help)[T.y] maybe, depending on category order
terms_ols += [t for t in ols.params.index if t.startswith('C(sex)') or t.startswith('C(help)')]

ols_terms = extract_terms(ols, terms_ols)
cluster_terms = extract_terms(cluster, terms_ols)
poisson_terms = extract_terms(poisson, terms_ols)

# Save results to csv for inspection
summary.to_csv('summary_stats.csv')
ols_terms.to_csv('ols_terms.csv', index=False)
cluster_terms.to_csv('cluster_terms.csv', index=False)
poisson_terms.to_csv('poisson_terms.csv', index=False)

# Print key outputs
print('N:', len(df))
print('\nEfficiency summary:\n', summary)
print('\nOLS (HC3) params:\n', ols_terms)
print('\nOLS cluster params:\n', cluster_terms)
print('\nPoisson (HC3) params:\n', poisson_terms)
print('\nOLS R2:', ols.rsquared)
print('OLS adj R2:', ols.rsquared_adj)
print('Poisson AIC:', poisson.aic)

# Group means for interpretability
group_sex = df.groupby('sex', observed=True)['efficiency'].agg(['mean', 'median', 'count'])
group_help = df.groupby('help', observed=True)['efficiency'].agg(['mean', 'median', 'count'])
print('\nEfficiency by sex:\n', group_sex)
print('\nEfficiency by help:\n', group_help)
