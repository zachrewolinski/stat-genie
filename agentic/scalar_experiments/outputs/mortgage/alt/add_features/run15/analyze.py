import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = 'mortgage.csv'

df = pd.read_csv(DATA_PATH)

print('rows', len(df))
print('columns', df.columns.tolist())

# Select relevant columns
cols = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI', 'accept'
]

missing_cols = [c for c in cols if c not in df.columns]
print('missing_cols', missing_cols)

use_cols = [c for c in cols if c in df.columns]
sub = df[use_cols].copy()

# Drop rows with missing values in selected columns
sub = sub.dropna()

print('rows_after_dropna', len(sub))

# Basic approval rates by gender
rate_by_gender = sub.groupby('female')['accept'].mean()
count_by_gender = sub.groupby('female')['accept'].count()
print('approval_rate_by_female')
print(rate_by_gender)
print('counts_by_female')
print(count_by_gender)

# Chi-square test of independence
cont_table = pd.crosstab(sub['female'], sub['accept'])
chi2, p, dof, expected = stats.chi2_contingency(cont_table)
print('chi2', chi2, 'p', p, 'dof', dof)
print('contingency_table')
print(cont_table)

def fit_logit(df, label):
    X = df.drop(columns=['accept'])
    X = sm.add_constant(X)
    y = df['accept']
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    print(f'\\n=== Logit: {label} ===')
    print(result.summary())
    coef = result.params.get('female')
    se = result.bse.get('female')
    z = coef / se if se is not None else np.nan
    pval = result.pvalues.get('female')
    or_val = np.exp(coef) if coef is not None else np.nan
    ci = result.conf_int().loc['female'] if 'female' in result.params.index else [np.nan, np.nan]
    ci_or = np.exp(ci)
    print('female_coef', coef)
    print('female_se', se)
    print('female_z', z)
    print('female_p', pval)
    print('female_or', or_val)
    print('female_or_ci', ci_or.tolist())
    margeff = result.get_margeff(at='mean')
    me = margeff.summary_frame().loc['female'] if 'female' in margeff.summary_frame().index else None
    print('marginal_effect_female')
    print(me)
    return result

# Logistic regression with full controls
fit_logit(sub, 'full_controls')

# Logistic regression without denied_PMI (potentially post-decision)
cols_no_pmi = [c for c in use_cols if c not in ['denied_PMI']]
sub_no_pmi = df[cols_no_pmi].dropna()
fit_logit(sub_no_pmi, 'no_denied_pmi')
