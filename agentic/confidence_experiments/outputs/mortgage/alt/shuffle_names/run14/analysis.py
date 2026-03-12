import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.width', 200)

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic info
print('rows', len(_df))
print('missing female', _df['female'].isna().sum())

# Crosstab accept vs deny
if 'accept' in _df.columns and 'deny' in _df.columns:
    ct = pd.crosstab(_df['accept'], _df['deny'], dropna=False)
    print('crosstab accept x deny\n', ct)
    print('accept+deny equals 1 proportion', ((_df['accept'] + _df['deny'])==1).mean())

# Outcome variable: use deny as denial indicator, or accept as approval indicator
# We'll compute approval rate by female using both
for outcome in ['accept', 'deny']:
    if outcome in _df.columns:
        print('\nOutcome:', outcome)
        grp = _df.groupby('female')[outcome].agg(['mean','count','sum'])
        print(grp)
        # Two-proportion z-test (male vs female)
        # female ==1 vs 0
        if _df['female'].nunique() == 2:
            g0 = _df[_df['female']==0][outcome].dropna()
            g1 = _df[_df['female']==1][outcome].dropna()
            # prop test
            p0 = g0.mean(); p1 = g1.mean()
            n0 = len(g0); n1 = len(g1)
            p_pool = (g0.sum() + g1.sum()) / (n0 + n1)
            se = np.sqrt(p_pool*(1-p_pool)*(1/n0 + 1/n1)) if n0>0 and n1>0 else np.nan
            z = (p1 - p0)/se if se>0 else np.nan
            pval = 2*(1-stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            print('prop test: p0(male)=%.4f p1(female)=%.4f z=%.3f p=%.4g' % (p0, p1, z, pval))

# Logistic regression: outcome as approval(accept) and female predictor
# We'll run simple logistic regression; also run with other covariates as sensitivity using available columns

# pick candidate covariates: exclude outcome, female
covariates = [c for c in _df.columns if c not in ['accept','deny','female']]
# drop obviously unique index-like column if it appears to be monotonic or high unique
# Identify columns with near-unique values
high_unique = [c for c in covariates if _df[c].nunique() > 0.95*len(_df)]
print('\nHigh-unique columns:', high_unique)

# Define function to run logit

def run_logit(outcome, covs):
    df = _df[[outcome,'female'] + covs].dropna()
    if df[outcome].nunique() < 2:
        print('Outcome has no variation for', outcome)
        return None
    formula = outcome + ' ~ female'
    if covs:
        formula += ' + ' + ' + '.join(covs)
    model = smf.logit(formula, data=df).fit(disp=False)
    print('\nLogit:', formula)
    print(model.summary().tables[1])
    return model

# Simple logit without covariates
run_logit('accept', [])
run_logit('deny', [])

# Logit with covariates excluding high_unique and nonbinary? We'll include numeric covariates
covs = [c for c in covariates if c not in high_unique]
run_logit('accept', covs)
run_logit('deny', covs)
