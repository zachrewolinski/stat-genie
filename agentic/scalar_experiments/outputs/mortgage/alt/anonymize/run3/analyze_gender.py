import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    'feature1': 'id',
    'feature2': 'female',
    'feature3': 'black',
    'feature4': 'housing_expense_ratio',
    'feature5': 'self_employed',
    'feature6': 'married',
    'feature7': 'mortgage_score',
    'feature8': 'consumer_score',
    'feature9': 'bad_credit',
    'feature10': 'debt_to_income',
    'feature11': 'denied',
    'feature12': 'loan_to_value',
    'feature13': 'pmi_denied',
    'feature14': 'accepted',
}

df = df.rename(columns=cols)

# Replace inf with NaN
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

# Basic sanity checks
# acceptance is expected to be the inverse of denied
if 'accepted' in df.columns and 'denied' in df.columns:
    mismatch = (df['accepted'] + df['denied']) != 1
    mismatch_count = int(mismatch.sum())
else:
    mismatch_count = None

# Acceptance rates by gender (drop rows missing female or accepted)
rate_df = df.dropna(subset=['female', 'accepted'])
summary = rate_df.groupby('female')['accepted'].agg(['count', 'mean'])

# Two-proportion test via chi-square on contingency table
contingency = pd.crosstab(rate_df['female'], rate_df['accepted'])
# ensure ordering: rows 0,1; columns 0,1
contingency = contingency.reindex(index=[0, 1], columns=[0, 1])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression: accepted ~ female
unadj_df = df.dropna(subset=['female', 'accepted'])
X_unadj = sm.add_constant(unadj_df['female'])
model_unadj = sm.Logit(unadj_df['accepted'], X_unadj).fit(disp=False)

# Adjusted logistic regression: accepted ~ female + covariates (exclude id, denied, accepted)
features = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_score', 'consumer_score', 'bad_credit', 'debt_to_income',
    'loan_to_value', 'pmi_denied'
]
adj_df = df.dropna(subset=features + ['accepted'])
X_adj = sm.add_constant(adj_df[features])
model_adj = sm.Logit(adj_df['accepted'], X_adj).fit(disp=False)

# Extract gender effect
coef_unadj = model_unadj.params['female']
ci_unadj = model_unadj.conf_int().loc['female']
coef_adj = model_adj.params['female']
ci_adj = model_adj.conf_int().loc['female']

# Convert to odds ratios
or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))
ci_or_unadj = tuple(np.exp(ci_unadj))
ci_or_adj = tuple(np.exp(ci_adj))

# Output results
print('mismatch_count', mismatch_count)
print('n_total', len(df))
print('n_rate', len(rate_df))
print('n_unadj', len(unadj_df))
print('n_adj', len(adj_df))
print('summary')
print(summary)
print('contingency')
print(contingency)
print('chi2', chi2, 'p', p_chi2)
print('unadj_coef', coef_unadj, 'p', model_unadj.pvalues['female'])
print('unadj_or', or_unadj, 'ci', ci_or_unadj)
print('adj_coef', coef_adj, 'p', model_adj.pvalues['female'])
print('adj_or', or_adj, 'ci', ci_or_adj)
