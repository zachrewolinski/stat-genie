import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Based on info.json descriptions (column names are shuffled)
# gender indicator: denied_PMI (1=female, 0=male)
# approval indicator: deny (1=accepted, 0=denied)
# denial indicator: self_employed (1=denied)

gender_col = 'denied_PMI'
approval_col = 'deny'

# Clean rows with missing in key columns
base = df[[gender_col, approval_col]].dropna()

# Group approval rates by gender
rates = base.groupby(gender_col)[approval_col].mean()
counts = base.groupby(gender_col)[approval_col].agg(['count','sum'])

# two-proportion z-test for approval difference
# group 1 = female (1), group 0 = male (0)
if 0 in rates.index and 1 in rates.index:
    n1 = counts.loc[1, 'count']
    n0 = counts.loc[0, 'count']
    p1 = rates.loc[1]
    p0 = rates.loc[0]
    p_pool = (counts.loc[1, 'sum'] + counts.loc[0, 'sum']) / (n1 + n0)
    z = (p1 - p0) / np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_z = np.nan

# chi-square test on contingency table
ct = pd.crosstab(base[gender_col], base[approval_col])
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)

# Logistic regression with controls
# Use all other columns except approval and gender.
# Exclude 'self_employed' because info.json says it is denial indicator (leakage).
exclude = {approval_col, gender_col, 'self_employed'}
# Drop index-like columns with too many unique values if they are just row IDs. 'bad_history' appears like index (0..2379)
# But we follow metadata: bad_history seems like index. We'll exclude it as likely identifier.
exclude.add('bad_history')

features = [c for c in df.columns if c not in exclude]

# Build formula
formula = approval_col + ' ~ ' + gender_col
if features:
    formula += ' + ' + ' + '.join(features)

# Drop NA rows for model
model_df = df[[approval_col, gender_col] + features].dropna()
model = smf.logit(formula, data=model_df).fit(disp=False)

# Extract gender effect
coef = model.params[gender_col]
se = model.bse[gender_col]
pval = model.pvalues[gender_col]
ci = model.conf_int().loc[gender_col].tolist()

# Output key stats
print('Gender col:', gender_col, 'Approval col:', approval_col)
print('Approval rate by gender (0=male,1=female):', rates.to_dict())
print('Counts by gender:', counts.to_dict())
print('Two-proportion z-test: z=%.4f p=%.4f' % (z, p_z))
print('Chi-square test: chi2=%.4f p=%.4f' % (chi2, p_chi))
print('Logit formula:', formula)
print('Logit gender coef=%.4f se=%.4f p=%.4f CI=%s' % (coef, se, pval, ci))
print('Odds ratio for female:', np.exp(coef), 'CI', np.exp(ci))
