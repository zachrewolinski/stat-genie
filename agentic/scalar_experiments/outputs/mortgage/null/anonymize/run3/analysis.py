import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic checks
# Define variables
outcome = 'feature14'  # accepted

gender = 'feature2'  # female indicator

# Candidate covariates (exclude outcome and its complement)
covariates = [
    'feature3',  # Black
    'feature4',  # housing expense / income
    'feature5',  # self-employed
    'feature6',  # married
    'feature7',  # mortgage credit score
    'feature8',  # consumer credit score
    'feature9',  # bad credit history
    'feature10', # debt-to-income
    'feature12', # loan-to-value
    'feature13', # PMI denied
]

# Drop rows with missing values in analysis columns
analysis_cols = [outcome, gender] + covariates
clean = df[analysis_cols].dropna().copy()

# Unadjusted approval rates by gender
rates = clean.groupby(gender)[outcome].agg(['mean', 'count'])

# Two-proportion z-test (female vs male)
# female=1, male=0
female = clean[clean[gender] == 1][outcome]
male = clean[clean[gender] == 0][outcome]

count = np.array([female.sum(), male.sum()])
obs = np.array([female.shape[0], male.shape[0]])
# Use statsmodels proportion ztest via scipy? We'll compute z-test manually
# Pooled proportion
p_pool = count.sum() / obs.sum()
se = np.sqrt(p_pool * (1 - p_pool) * (1/obs[0] + 1/obs[1]))
z = (female.mean() - male.mean()) / se if se > 0 else np.nan
p_unadj = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

# Logistic regression adjusted
X = clean[[gender] + covariates]
X = sm.add_constant(X, has_constant='add')
y = clean[outcome]

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

coef = res.params[gender]
se_coef = res.bse[gender]
odds_ratio = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se_coef))
ci_high = float(np.exp(coef + 1.96 * se_coef))
p_adj = float(res.pvalues[gender])

# McFadden pseudo R2 for reference
ll_null = res.llnull
ll_model = res.llf
pseudo_r2 = 1 - ll_model/ll_null if ll_null != 0 else np.nan

# Output summary for human reading (printed)
print('N', len(clean))
print('Approval rate female', female.mean(), 'n', obs[0])
print('Approval rate male', male.mean(), 'n', obs[1])
print('Unadjusted diff (female-male)', female.mean() - male.mean())
print('Unadjusted z', z, 'p', p_unadj)
print('Adjusted OR (female vs male)', odds_ratio)
print('95% CI', (ci_low, ci_high))
print('Adjusted p', p_adj)
print('Pseudo R2', pseudo_r2)
